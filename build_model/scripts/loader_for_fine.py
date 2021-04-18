import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras import backend as K





if __name__=='__main__':
	# Open Data
	THREE = True
	NINE = True	

	import sys

	if True:
		print('\n\n\n\nUSING UPGRADED DATA!\n\n\n')
		#BASE_DIR = 'npdata/lap2-rand-con-empates'
		BASE_DIR='.'
		X=np.load(BASE_DIR + '/'+'x.npy')#[:10000]
		Y=np.load(BASE_DIR + '/'+'y.npy')#[:10000]*-1
	else:
		X=np.load('xfine.npy')
		Y=np.load('yfine.npy')

	L = int(0.95*len(X))
	CW = {i:x for i,x in enumerate(L/np.sum(Y[:L],0)/Y.shape[1])}
	print(CW)

	try:
		base_model = models.load_model('model-onlywin-fine.h5')
	except:
		base_model = models.load_model('model-onlywin.h5')
	
	new_model = models.Model(base_model.input, outputs=base_model.output)

	for i in range(len(base_model.layers)): 
		layers.trainable = True   # True--> fine tine, False-->frozen

	print("Number of layers in the new model: ", len(new_model.layers))

	try:
		lr = float(sys.argv[3])
		print(f'\n\n\nRECIEVED LEARNING RATE: {lr}\n\n\n')
	except: lr = 0.001
	new_model.compile(optimizer=optimizers.Adam(lr=lr),
		          loss='categorical_crossentropy',
		          metrics=['accuracy'])

	tb_dir = "./fine-tune-logs"
	tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
		                                     write_graph=True)

	if len(X)<100:
		history = new_model.fit(X,      # input your new training data and labels
		                Y,
		                batch_size=4,
		                epochs=20,
		                verbose=2,
		                #validation_data=(x_val, y_val),
		                callbacks=[tb_callback],
				)#class_weight=CW)
	else:
		try: 
			epochs = int(sys.argv[1])
			batch = int(sys.argv[2])
			print(f'\n\n\nRECIEVED BATCH AND EPOCHS: {batch}   {epochs}\n\n\n')
		except: 
			epochs = 30
			batch = 128
		history = new_model.fit(X,      # input your new training data and labels
		                Y,
		                batch_size=batch,
		                epochs=epochs,
		                verbose=1,
		                #validation_data=(x_val, y_val),
		                callbacks=[tb_callback],
				)#class_weight=CW)


	new_model.save('model-onlywin-fine.h5', save_format="h5")


