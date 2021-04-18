import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, recall_score
import loader
from scipy.stats import multinomial

from tensorflow.keras.models import load_model

zeros=np.zeros((1,9,9))
def move(history, markov=False):
	local = np.asarray([[x for x in y] for y in history])
	if markov:
		print('\n      (markovian)     \n')
		x=np.concatenate([zeros[:,:len(local)-1,:],np.asarray(local)[-1:].reshape(1,1,9), zeros[:,len(local):,:]],1)#.reshape(1,9,9)
	else:
		x=np.concatenate([np.asarray(local).reshape(1,len(local),9), zeros[:,len(local):,:]],1).reshape(1,9,9)
	print(x)
	result = model.predict(x)
	return result




def board_printer(board):
	title='Current Board'
	L = int(np.sqrt(len(board)))
	filler = '.'*int(((2+5*L)-len(f'{title}'))/2)
	underline = '_'*5*L + '_.'
	q=f'{filler}{title}{filler}\n{underline}\n'
	counter=1
	for x in board:
		if counter>3:
			counter=1
			q+='\n|'
		else:
			q += '|'
		if x==0: q += '|___|'
		elif x==-1: q += '|_X_|'
		elif x==1: q += '|_O_|'
		counter+=1     
	q+='\n\n\n'
	print(q)




if __name__=='__main__':
	markov=False
	history=[]
	DETERMINISTIC=True
	try:
		import sys
		model = load_model(sys.argv[1]+'.h5')
		print('recieved ',sys.argv[1])
	except:
		model = load_model('model-onlywin-fine'+'.h5')
	board = np.zeros(9)
	board_printer(board)
	first = True

	print(f'Deterministic: {DETERMINISTIC}\nMarkovian:{markov}\n--------------	')

	while True:
		row = int(input('row (1-3): '))-1
		col = int(input('col (1-3): '))-1
		board[row*3+col] = 1
		history.append(board.copy())
		board_printer(board)
		probas = move(history, markov=markov)
		print(f'probas are: {probas}')
		if DETERMINISTIC:
			probas = {v:k for k,v in enumerate(probas[0]) if board[k]==0} 
			IX = probas[max(list(probas.keys()))]
		else:
			probas = [x if board[i]==0 else 0 for i,x in enumerate(probas[0])]
			freexs = [i for i,x in enumerate(probas) if x>0]
			s=sum(probas)
			IX = freexs[np.argmax(multinomial.rvs(1,p=[x/s for x in probas if x>0]))]
		print(f'processed probas are: {probas}')
		board[IX] = -1
		board_printer(board)
		history.append(board.copy())
		



	
