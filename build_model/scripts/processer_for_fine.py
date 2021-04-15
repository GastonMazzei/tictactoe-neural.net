import pickle
import tensorflow as tf
import numpy as np


# 
# Produces a dataset that has:
#           (board1, board2, board3,...)   --> (3,0,0)
#           (board1, board2, board3,...)   --> (0,2,0)
#



errors = []
err = 0

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

with open('enhaceresults.pkl','rb') as f:
  data = pickle.load(f)

k = list(data.keys())

base = (-1, -1, -1, -1, -1, -1, -1, -1, -1)

cases = {x:{} for x in [0]}
zero = (0,)*9


for i in cases.keys():
  for k_ in k:
    try:
      localkey =  tuple(tuple(y for y in x) for x in k_)+(zero,)*(9-len(k_))  #k_+(zero,)*(9-len(k_))
      if localkey in cases[i]:
        cases[i][localkey] = [cases[i][localkey][j]+data[k_][j] for j in range(3)]
      else:
        cases[i][localkey] = data[k_]
    except Exception as ins:
      err += 1
      errors.append(ins.args)

for k in cases.keys():
  localkeys = list(cases[k].keys())
  localx = []
  localy = []
  for k_ in localkeys:
    localx.append(k_)
    localy.append(cases[k][k_]) 
  x=np.asarray(localx)
  print(f'Xs shape is {x.shape}')
  y=np.asarray(localy).reshape(-1,3)
  x,y = unison_shuffled_copies(x, y)


print('BEFORE:')
print(x.shape)
print(y.shape)	
print('AFTER:')
ixs = [i for i,z in enumerate(y) if sum(z[-2:])>0]#z[-1]>0]#sum(z[-2:])>0]
x = x[ixs]
y = y[ixs]
z0 = np.zeros(9)
z = np.zeros((9,9))

xnew, ynew = [],[]
for x_ in x:
	try:
		for i in range(9):
			if sum(x_[i,:]==z0)==9:
				j=i
				break
		xnew += [np.concatenate([x_[:i,:],z[i:,:]],0) for i in range(1,j) if i%2==1]
		ynew += [x_[i-1,:] - x_[i,:] for i in range(1,j) if i%2==1]
	except Exception as ins: 
		errors.append(ins.args)
		err += 1

ynew = np.concatenate(ynew,0).reshape(-1,9)
xnew = np.concatenate(xnew,0).reshape(-1,9,9)
print(xnew.shape)
print(ynew.shape)	

np.save('xfine.npy',xnew)
np.save('yfine.npy',ynew)





errors = []
err = 0

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

with open('enhaceresults.pkl','rb') as f:
  data = pickle.load(f)

k = list(data.keys())

base = (-1, -1, -1, -1, -1, -1, -1, -1, -1)

cases = {x:{} for x in [0]}
zero = (0,)*9


for i in cases.keys():
  for k_ in k:
    try:
      localkey =  tuple(tuple(y for y in x) for x in k_)+(zero,)*(9-len(k_))  #k_+(zero,)*(9-len(k_))
      if localkey in cases[i]:
        cases[i][localkey] = [cases[i][localkey][j]+data[k_][j] for j in range(3)]
      else:
        cases[i][localkey] = data[k_]
    except Exception as ins:
      err += 1
      errors.append(ins.args)

for k in cases.keys():
  localkeys = list(cases[k].keys())
  localx = []
  localy = []
  for k_ in localkeys:
    localx.append(k_)
    localy.append(cases[k][k_]) 
  x=np.asarray(localx)
  print(f'Xs shape is {x.shape}')
  y=np.asarray(localy).reshape(-1,3)
  x,y = unison_shuffled_copies(x, y)


print('BEFORE:')
print(x.shape)
print(y.shape)	
print('AFTER:')
ixs = [i for i,z in enumerate(y) if sum(z[:2])>0]#z[0]>0]#sum(z[:2])>0]
x = x[ixs]
y = y[ixs]
z0 = np.zeros(9)
z = np.zeros((9,9))

xnew, ynew = [],[]
for x_ in x:
	try:
		for i in range(9):
			if sum(x_[i,:]==z0)==9:
				j=i
				break 
		xnew += [np.concatenate([x_[:i,:],z[i:,:]],0) for i in range(1,j) if i%2==0]
		ynew += [x_[i-1,:] - x_[i,:] for i in range(1,j) if i%2==0]
	except Exception as ins: 
		errors.append(ins.args)
		err += 1

ynew = np.concatenate(ynew,0).reshape(-1,9)
xnew = np.concatenate(xnew,0).reshape(-1,9,9)
print(xnew.shape)
print(ynew.shape)	

np.save('x2fine.npy',xnew)
np.save('y2fine.npy',ynew)






