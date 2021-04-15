import numpy as np










if __name__=='__main__':
	x=np.load('x.npy.old')
	y=np.load('y.npy.old')
	print('BEFORE:')
	print(x.shape)
	print(y.shape)	
	print('AFTER:')
	ixs = [i for i,z in enumerate(y) if sum(z[:2])>0]
	x = x[ixs]
	y = y[ixs]
	z0 = np.zeros(9)
	z = np.zeros((9,9))

	xnew, ynew = [],[]
	for x_ in x:
		for i in range(9):
			if sum(x_[i,:]==z0)==9:
				j=i
				break 
		xnew += [np.concatenate([x_[:i,:],z[i:,:]],0) for i in range(1,j) if i%2==0]
		ynew += [x_[i-1,:] - x_[i,:] for i in range(1,j) if i%2==0]

	ynew = np.concatenate(ynew,0).reshape(-1,9)
	xnew = np.concatenate(xnew,0).reshape(-1,9,9)
	print(xnew.shape)
	print(ynew.shape)	

	np.save('x2.npy',xnew)
	np.save('y2.npy',ynew)



