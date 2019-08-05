import numpy as np

def dec_to_bin(x, ndigits):
	b = np.zeros((ndigits,), dtype = int)
	for i in range(0, ndigits):
		q, r = divmod(x, 2)
		b[ndigits-i-1] = r
		x = q
	return b

def dec_to_trin(x, ndigits):
	b = np.zeros((ndigits,), dtype=int)
	for i in range(0, ndigits):
		q, r = divmod(x, 3)
		b[ ndigits-i-1] = r
		x = q
	return b

def colProduct(M1, M2 = None):
	
	if M2 is None:
		nrow, ncol = M1.shape
		R = np.zeros( (nrow, ncol*(ncol+1)/2), dtype=float )
		k = 0
		for i in range(ncol):
			for j in range(i,ncol):
				R[:,k] = M1[:,i] * M1[:,j]
				k += 1
	else:
		nrow1, ncol1 = M1.shape
		nrow2, ncol2 = M2.shape
		R = np.zeros( (nrow1, ncol1*ncol2) )
		k = 0
		for i in range(ncol1):
			for j in range(ncol2):
				R[:,k] = M1[:,i] * M2[:,j]
				k += 1
	return R

if __name__ == '__main__':
	print dec_to_bin(2,4)