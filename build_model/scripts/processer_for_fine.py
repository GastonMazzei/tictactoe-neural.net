import pickle
import tensorflow as tf
import numpy as np



IN = 'results'
OUT = '' 
# 
# Produces a dataset that has:
#           (board1, board2, board3,...)   --> (3,0,0)
#           (board1, board2, board3,...)   --> (0,2,0)
#

import sys
if len(sys.argv)>1:
  SINGLE = bool(int(sys.argv[1]))
  TIES = bool(int(sys.argv[2]))
  print(f'single? {SINGLE}, ties? {TIES}')
else:
  SINGLE=False
  TIES = True
  ONLYTIES = False

errors = []
err = 0

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

with open(f'{IN}.pkl','rb') as f:
  data = pickle.load(f)

k = list(data.keys())

base = (-1, -1, -1, -1, -1, -1, -1, -1, -1)

cases = {x:{} for x in [0]}
zero = (0,)*9


for i in cases.keys():
  for k_ in k:  
    try:
#      localkey =  tuple(tuple(-1-y if y in [-1,0] else y for y in x) for x in k_)+(zero,)*(9-len(k_))  #k_+(zero,)*(9-len(k_))
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

# condition: include ties or not? do it N times or not?
if TIES:
  if ONLYTIES:
    ixs = [i for i,z in enumerate(y) if z[1]>0]
  else:
    ixs = [i for i,z in enumerate(y) if sum(z[-2:])>0]
  if SINGLE: 
    times = [1 for i,z in enumerate(y) if sum(z[-2:])>0]
  else:
    if ONLYTIES:
      times = [z[1] for i,z in enumerate(y) if z[1]>0]
    else:
      times = [z[-2]+z[-1] for i,z in enumerate(y) if sum(z[-2:])>0]
else:
  ixs = [i for i,z in enumerate(y) if z[-1]>0]#z[-1]>0]#sum(z[-2:])>0]
  if SINGLE: 
    times = [1 for i,z in enumerate(y) if z[-1]>0]
  else:
    times = [z[-1] for i,z in enumerate(y) if z[-1]>0]


x = x[ixs]
y = y[ixs]
z0 = np.zeros(9)
z = np.zeros((9,9))

xnew, ynew = [],[]
for l,x_ in enumerate(x):
	xnew_local, ynew_local = [],[]
	try:
		for i in range(9):
			if (sum(x_[i,:]==z0)==9) or (i==8):
				j=i
				break
		xnew_local += [np.concatenate([x_[:i,:],z[i:,:]],0) for i in range(1,j) if i%2==1]
		ynew_local += [x_[i-1,:] - x_[i,:] for i in range(1,j) if i%2==1]
	except Exception as ins: 
		errors.append(ins.args)
		err += 1
	xnew += (xnew_local * times[l]).copy()
	ynew += (ynew_local * times[l]).copy()

if len(errors)>0: 
  print('errors: ',err)
  print(errors[0])

ynew = np.abs(np.concatenate(ynew,0).reshape(-1,9))
xnew = np.concatenate(xnew,0).reshape(-1,9,9)


xnew,ynew = unison_shuffled_copies(xnew, ynew)

print(xnew.shape)
print(ynew.shape)	

np.save(f'x{OUT}.npy',xnew)
np.save(f'y{OUT}.npy',ynew)





errors = []
err = 0

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

with open(f'{IN}.pkl','rb') as f:
  data = pickle.load(f)

k = list(data.keys())

base = (-1, -1, -1, -1, -1, -1, -1, -1, -1)

cases = {x:{} for x in [0]}
zero = (0,)*9


for i in cases.keys():
  for k_ in k:
    try:
#      localkey =  tuple(tuple(-1-y if y in [-1,0] else y for y in x) for x in k_)+(zero,)*(9-len(k_))  #k_+(zero,)*(9-len(k_))
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

# condition: include ties or not? do it N times or not?

if TIES:
  if ONLYTIES:
    ixs = [i for i,z in enumerate(y) if z[1]>0]
  else:
    ixs = [i for i,z in enumerate(y) if sum(z[:2])>0]
  if SINGLE: 
    times = [1 for i,z in enumerate(y) if sum(z[:2])>0]
  else:
    if ONLYTIES:
      times = [z[1] for i,z in enumerate(y) if z[1]>0]
    else:
      times = [z[1]+z[0] for i,z in enumerate(y) if sum(z[:2])>0]
else:
  ixs = [i for i,z in enumerate(y) if z[0]>0]#z[-1]>0]#sum(z[-2:])>0]
  if SINGLE: 
    times = [1 for i,z in enumerate(y) if z[0]>0]
  else:
    times = [z[0] for i,z in enumerate(y) if z[0]>0]


x = x[ixs]
y = y[ixs]
z0 = np.zeros(9)
z = np.zeros((9,9))




xnew, ynew = [],[]
for l,x_ in enumerate(x):
	xnew_local, ynew_local = [],[]
	try:
		for i in range(9):
			if (sum(x_[i,:]==z0)==9) or (i==8):
				j=i
				break
		xnew_local += [np.concatenate([x_[:i,:],z[i:,:]],0) for i in range(1,j) if i%2==0]
		ynew_local += [x_[i-1,:] - x_[i,:] for i in range(1,j) if i%2==0]
	except Exception as ins: 
		errors.append(ins.args)
		err += 1
	xnew += (xnew_local * times[l]).copy()
	ynew += (ynew_local * times[l]).copy()

if len(errors)>0: 
  print('errors: ',err)
  print(errors[0])

ynew = np.abs(np.concatenate(ynew,0).reshape(-1,9))
xnew = np.concatenate(xnew,0).reshape(-1,9,9)


xnew,ynew = unison_shuffled_copies(xnew, ynew)

print(xnew.shape)
print(ynew.shape)


	
np.save(f'x2{OUT}.npy',xnew)
np.save(f'y2{OUT}.npy',ynew)







