import numpy as np

FINE = False

BASE_DIR = 'npdata/rand-sin-empates'
DEST_DIR = 'npdata/upgrade-rand-sin-empates'

if FINE:
  x,y = np.load(BASE_DIR + '/'+ 'xfine.npy'),np.load(BASE_DIR + '/'+ 'yfine.npy')
else:
  x,y = np.load(BASE_DIR + '/'+ 'x.npy'),np.load(BASE_DIR + '/'+ 'y.npy')


print(f'INITIAL LENGTH: {len(x)}')

# Starting in the middle
a=np.asarray([[0,0,0,0,1,0,0,0,0],
*[np.asarray([0]*9)]*8])

# Starting in another place?
a2=np.asarray([[0,0,0,0,1,0,0,0,0],
*[np.asarray([0]*9)]*8])


cases = [i for i in range(len(x)) if np.sum(x[i,0,:]==a[0,:])==9]
#cases += [i for i in range(len(x)) if np.sum(x[i,0,:]==a2[0,:])==9]


def view(v,a):
  q = v[a].copy()
  r = [['']*9 for _ in range(9)]
  for i,x in enumerate(q):
    if np.sum(x)>0:
      for j,y in enumerate(x):
        r[i][j] = ' ' if y==0 else ('O' if y==1 else 'X')
      for k in range(3):
        print(r[i][k*3:k*3+3])    
      print('\n\n')
  
IX = cases[1]+2

view(x, IX)

print(y[IX])

print(f'FINAL LENGTH: {len(cases)}')

np.save(DEST_DIR + '/' + 'x.npy',x[cases])
np.save(DEST_DIR + '/' + 'y.npy',y[cases])




if FINE:
  x,y = np.load(BASE_DIR + '/'+ 'x2fine.npy'),np.load(BASE_DIR + '/'+ 'y2fine.npy')
else:
  x,y = np.load(BASE_DIR + '/'+ 'x2.npy'),np.load(BASE_DIR + '/'+ 'y2.npy')

print(f'INITIAL LENGTH: {len(x)}')

# Starting in the middle
a=np.asarray([[0,0,0,0,1,0,0,0,0],
*[np.asarray([0]*9)]*8])

# Starting in another place?
a2=np.asarray([[0,0,0,0,1,0,0,0,0],
*[np.asarray([0]*9)]*8])


cases = [i for i in range(len(x)) if np.sum(x[i,0,:]==a[0,:])==9]
#cases += [i for i in range(len(x)) if np.sum(x[i,0,:]==a2[0,:])==9]


def view(v,a):
  q = v[a].copy()
  r = [['']*9 for _ in range(9)]
  for i,x in enumerate(q):
    if np.sum(x)>0:
      for j,y in enumerate(x):
        r[i][j] = ' ' if y==0 else ('O' if y==1 else 'X')
      for k in range(3):
        print(r[i][k*3:k*3+3])    
      print('\n\n')
  
IX = cases[1]+2

view(x, IX)

print(y[IX])

np.save(DEST_DIR + '/' + 'x2.npy',x[cases])
np.save(DEST_DIR + '/' + 'y2.npy',y[cases])


print(f'FINAL LENGTH: {len(cases)}')

























