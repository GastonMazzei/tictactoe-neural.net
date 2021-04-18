import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
import sys



def create_model(s,NEUR=8, name="LSTM_chess_model"):
	inputs = tf.keras.Input(shape=s)
	lstm = tf.keras.layers.LSTM(NEUR, )#return_sequences=False, return_state=False)
	x = lstm(inputs)
	x = tf.keras.layers.Dense(NEUR, activation="relu")(x)
	x = tf.keras.layers.Dense(NEUR, activation="relu")(x)
	x = tf.keras.layers.Dense(NEUR, activation="relu")(x)
	x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
	outputs = (x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
	print(model.summary())
	return model

def create_model_three_classes(s,NEUR=8, name="LSTM_chess_model"):
	inputs = tf.keras.Input(shape=s)
	lstm = tf.keras.layers.LSTM(NEUR, return_state=True)#return_sequences=False, return_state=False)
	x = lstm(inputs)
	x = tf.keras.layers.Dense(NEUR, activation="relu")(x)
	#x = tf.keras.layers.Dropout(0.1)(x)
	x = tf.keras.layers.Dense(NEUR, activation="relu")(x)
	#x = tf.keras.layers.Dropout(0.1)(x)
	x = tf.keras.layers.Dense(3, activation="softmax")(x)
	outputs = (x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
	print(model.summary())
	return model

def create_model_nine_classes(s,NEUR=8, name="LSTM_chess_model", ACT='tanh'):
	inputs = tf.keras.Input(shape=s)
	lstm = tf.keras.layers.LSTM(NEUR, )#return_sequences=False, return_state=False)
	x = lstm(inputs)
	x = tf.keras.layers.Dense(NEUR, activation=ACT)(x)#"relu")(x)
	x = tf.keras.layers.Dense(NEUR, activation=ACT)(x)#"relu")(x)
	x = tf.keras.layers.Dense(NEUR, activation=ACT)(x)#"relu")(x)
	x = tf.keras.layers.Dense(9, activation="softmax")(x)
	outputs = (x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
	print(model.summary())
	return model


def energy_to_proba(Y_continuous, beta=0.5, binary=False, threshold=0.5, view=False):
	E_won = -1
	E_tie =  0
	E_lost = 1
	Y_shape = (len(Y_continuous),1)
	s = np.sum(Y_continuous,1)
	if  False:
		Y = ( #(Y_continuous[:,0]/s) * np.exp(-1*beta*E_lost) + <--This is not included, as E(p)= 0*P(0) + 1*P(1)
			(Y_continuous[:,1]/s) * np.exp(-1*beta*E_tie) +
			(Y_continuous[:,2]/s) * np.exp(-1*beta*E_won) ) / ( sum([np.exp(-1*beta*E) for E in [E_won, E_lost, E_tie]]) )
	else:
		Y = np.where(np.argmax(Y_continuous,1)==1,1,0)
	if view:
		f,ax = plt.subplots(1,2,figsize=(10,5))
		ax[1].hist(Y.flatten(), bins=np.linspace(np.min(Y), np.max(Y), 30))
		ax[0].hist(Y_continuous.flatten(), bins=np.linspace(np.min(Y_continuous), np.max(Y_continuous), 30))
		ax[0].set_title('original energy')
		ax[1].set_title('proba after the boltzmann distribution')
		plt.show()
	if binary:
		return np.where(Y>threshold,1,0).reshape(Y_shape)
	return Y.reshape(Y_shape)

def energy_to_three_classes(Y_continuous, beta=0.5, binary=False, threshold=0.5, view=False):
	Y_shape = (len(Y_continuous),1)
	s = np.sum(Y_continuous,1)
	Y = np.argmax(Y_continuous,1)
	print(max(Y))
	return Y.reshape(Y_shape)


if __name__=='__main__':
	# Open Data
	THREE = True
	NINE = True	
	X=np.load('x.npy')
	Y = np.load('y.npy')
	L = int(0.95*len(X))

	print(f'recieved: {sys.argv}')
	try:
		model = create_model_nine_classes((X.shape[1],X.shape[2]), NEUR=int(sys.argv[1]), name='LSTM_tictactoe_model')
	except:
		model = create_model_nine_classes((X.shape[1],X.shape[2]), NEUR=64, name='LSTM_tictactoe_model')

	LOSS='categorical_crossentropy'
	print('\n\n\n\ncase with nine outputs...\n\n\n\n')
	Y=np.load('y.npy')
	CW = {i:x for i,x in enumerate(L/np.sum(Y[:L],0)/Y.shape[1])}


	# Train the model and save
	
	

	print(CW)
	opt = tf.keras.optimizers.Adam(learning_rate=0.03)
	model.compile(loss=LOSS, optimizer=opt, metrics='accuracy')
	try:
		result = model.fit(X[:L],Y[:L], epochs=int(sys.argv[2]), batch_size=int(sys.argv[3]),
                            validation_data=(X[L:],Y[L:]))#,  class_weight=CW)
	except:
		result = model.fit(X[:L],Y[:L], epochs=30, batch_size=20000,
                            validation_data=(X[L:],Y[L:]))#,  class_weight=CW)

	if not THREE:
		result.history['train_ROC'] = roc_curve(Y[:L], model.predict(X[:L]).flatten())
		result.history['val_ROC'] = roc_curve(Y[L:], model.predict(X[L:]).flatten())
	with open('resultsNN.pkl','wb') as f:
		pickle.dump(result.history,f)
	model.save('model-result.h5', save_format="h5")


