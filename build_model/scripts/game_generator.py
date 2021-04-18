from os import supports_dir_fd
import matplotlib.pyplot as plt
import pickle, sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, recall_score

from scipy.stats import multinomial

zeros=np.zeros((1,9,9))
def move(history, markov=False, O=False):
	local = np.asarray([[x for x in y] for y in history])
	if markov:
		x=np.concatenate([zeros[:,:len(local)-2,:],np.asarray(local)[-2:].reshape(1,2,9), zeros[:,len(local):,:]],1).reshape(1,9,9)
	else:
		x=np.concatenate([np.asarray(local).reshape(1,len(local),9), zeros[:,len(local):,:]],1).reshape(1,9,9)
	if O:
		result = model2.predict(x)
	else:
		result = model1.predict(x)
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

def check_winner(v):
	if len([x for x in v if x==0])==0:
		return True, [0,1,0]
	patterns = []
	patterns.append(v[6] + v[4] + v[2])
	patterns.append(v[0]+v[4]+v[8])
	patterns.append(sum(v[:3]))
	patterns.append(sum(v[3:6]))
	patterns.append(sum(v[6:9]))
	patterns.append(sum([v[0+i*3] for i in [0,1,2]]))
	patterns.append(sum([v[1+i*3] for i in [0,1,2]]))
	patterns.append(sum([v[2+i*3] for i in [0,1,2]]))
	if 3 in patterns:
		return True, [1,0,0]
	elif -3 in patterns:
		return True, [0,0,1]
	return False, []


if __name__=='__main__':

	PROBA=0
	try:
		print(sys.argv)
		VERBOSE, SINGLE, RANDOM = [bool(int(x)) for x in sys.argv[1:4]]
		print([bool(x) for x in sys.argv[1:4]])
	except:
		VERBOSE=True
		SINGLE = True
		RANDOM = False
		
	print(f'VERBOSE:{VERBOSE}\nSINGLE:{SINGLE}\nRANDOM:{RANDOM}')
	#model = tf.keras.models.load_model('model-result-20210413T004221Z-001/model-result')

	if len(sys.argv)>5:		
		model1 = load_model('model-onlywin-fine.h5')
		model2 = load_model('model-onlywin2-fine.h5')
		print('\n\n\nUSING FINE-TUNNED MODELS!\n\n\n\n')
	else:
		model1 = load_model('model-onlywin.h5')
		model2 = load_model('model-onlywin2.h5')
	try:
		N = int(sys.argv[4])
	except:
		N=1
	totalhistory = {}
	totalresult = []
	for _ in range(N):
		history=[]
		board = np.zeros(9)
		if  VERBOSE: board_printer(board)
		i=0
		while True:
			if  i==0 or SINGLE:
				ix = np.random.choice([i for i,x in enumerate(board) if x==0])
				board[ix] = 1
			else:
				probas = move(history, markov=False, O=True)
				if  RANDOM: # RANDOM could be set to false... 1rst player should be SMART! :-D (ad-hoc strategy)
					probas = [x if board[i]==0 else 0 for i,x in enumerate(probas[0])]
					freexs = [i for i,x in enumerate(probas) if x>0]
					s=sum(probas)
					IX = freexs[np.argmax(multinomial.rvs(1,p=[x/s for x in probas if x>0]))]
					try:
						pred = multinomial.rvs(1,p=[x/s for x in probas if x>0])
						if np.random.rand()>PROBA:
							IX = freexs[np.argmax(pred)]
						else:
							IX = np.random.choice(freexs)
					except:
						IX = np.random.choice([i for i,x in enumerate(board) if x==0])
						board[ix] = 1
						if len(history)>7: pass
						else: raise Exception(f'Error! length was: {len(history)}')
	#					import sys
	#					print('history is :\n\n',np.concatenate(history,0).reshape(-1,9),'\n')
	#					print('board is :\n\n',board,'\n')
	#					np.save('check.npy',np.concatenate(history,0).reshape(-1,9))
	#					np.save('checkboard.npy',np.asarray(probas))
	#					print(probas, np.argmax(pred)); sys.exit(1)
				else: #DETERMINISTIC
					probas = {v:k for k,v in enumerate(probas[0]) if board[k]==0} 
					IX = probas[max(list(probas.keys()))]
				
				board[IX] = 1
				if  VERBOSE:
					print(f'probas are: {probas}')
					print(f'chosen one is: {IX}')
			if VERBOSE:
				board_printer(board)

			i+=1
			history.append(board.copy())
			if i>=3:
				win,who  = check_winner(board)
				if win: 
					break
			probas = move(history, markov=False, O=False)
			if  RANDOM: # RANDOM
				probas = [x if board[i]==0 else 0 for i,x in enumerate(probas[0])]
				freexs = [i for i,x in enumerate(probas) if x>0]
				s=sum(probas)
				pred = multinomial.rvs(1,p=[x/s for x in probas if x>0])
				if np.random.rand()>PROBA:
					IX = freexs[np.argmax(pred)]
				else:
					IX = np.random.choice(freexs)

			else: #DETERMINISTIC
				probas = {v:k for k,v in enumerate(probas[0]) if board[k]==0} 
				IX = probas[max(list(probas.keys()))]

			board[IX] = -1
			if  VERBOSE:
				print(f'probas are: {probas}')
				print(f'chosen one is: {IX}')
				board_printer(board)
			history.append(board.copy())
			if i>=3:
				win,who  = check_winner(board)
				if win:
					break
		key = tuple(tuple(x) for x in history)
		if key in totalhistory:
			totalhistory[key] = [totalhistory[key][i]+who[i] for i in range(3)]
		else:
			totalhistory[key] = who.copy()

	#for x in totalhistory.keys():	
	#	print(x, totalhistory[x])

	with open('enhaceresults.pkl','wb') as f:
		pickle.dump(totalhistory,f)

	v = np.asarray(list(totalhistory.values()))
	print(round(100*sum(v[:,0])/len(v),1),'%',
		round(100*sum(v[:,1])/len(v),1),'%',
		round(100*sum(v[:,2])/len(v),1),'%')


	
