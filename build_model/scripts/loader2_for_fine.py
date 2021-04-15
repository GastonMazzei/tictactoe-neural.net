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
	X=np.load('x2fine.npy')
	L = int(0.8*len(X))
	Y=np.load('y2fine.npy')*-1
	CW = {i:x for i,x in enumerate(L/np.sum(Y[:L],0)/Y.shape[1])}
	print(CW)

	try:
		base_model = models.load_model('model-onlywin2-fine.h5')
	except:
		base_model = models.load_model('model-onlywin2.h5')
	new_model = models.Model(base_model.input, outputs=base_model.output)

	for i in range(len(base_model.layers)): 
		layers.trainable = True   # True--> fine tine, False-->frozen

	print("Number of layers in the new model: ", len(new_model.layers))

	new_model.compile(optimizer=optimizers.Adam(lr=0.0003),
		          loss='categorical_crossentropy',
		          metrics=['accuracy'])

	tb_dir = "./fine-tune-logs"
	tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
		                                     write_graph=True)

	history = new_model.fit(X,      # input your new training data and labels
		                Y,
		                batch_size=32,
		                epochs=60,
		                verbose=2,
		                #validation_data=(x_val, y_val),
		                callbacks=[tb_callback],
				)#class_weight=CW)

	new_model.save('model-onlywin2-fine.h5', save_format="h5")

	import matplotlib.pyplot as plt
	#plt.plot(history.history['loss']);plt.show()

