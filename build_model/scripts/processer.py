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

with open('results.pkl','rb') as f:
  data = pickle.load(f)

k = list(data.keys())

base = (-1, -1, -1, -1, -1, -1, -1, -1, -1)

cases = {x:{} for x in [0]}
zero = (0,)*9


for i in cases.keys():
  for k_ in k:
    try:
      localkey =  tuple(tuple(-1-y if y in [-1,0] else y for y in x) for x in k_)+(zero,)*(9-len(k_))  #k_+(zero,)*(9-len(k_))
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
  np.save('x.npy.old', x) 
  np.save('y.npy.old', y) 
  print('success!')




import sys
print(f'ENDED with {err} errors, saving them in LOGS')
with open('logs','w') as f:
  for x in errors:
    f.write(x+'\n')
sys.exit(1)





