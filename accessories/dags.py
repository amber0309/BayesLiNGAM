import numpy as np
from itertools import permutations
from scipy.misc import comb

import sys
sys.path.insert(1, '../accessories') # add the path of the auxiliary functions
import util

def all_bin_vecs(arr, v):
	"""
	create an array which holds all 2^V binary vectors

	INPUT 
	arr 		 positive integers from 1 to 2^V, (2^V, ) numpy array
	v 			 number of variables V
	OUTPUT
	edgeconfs 	 all binary vectors, (2^V, V) numpy array
	"""
	to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(v))
	strs = to_str_func(arr)
	edgeconfs = np.zeros((arr.shape[0], v), dtype=np.int8)
	for bit_ix in range(0, v):
		fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
		edgeconfs[:,bit_ix] = fetch_bit_func(strs)[:,0]

	return edgeconfs

def allfamilies(v):
	"""
	create two arrays which hold all possible families of v nodes

	INPUT
	v 			 number of variables v
	OUTPUT
	nodes 		 node index, (v*2^(v-1), ) numpy array
	pars 		 parents indices, (v*2^(v-1), v) numpy array
	"""

	edgeconfs = all_bin_vecs( np.arange(2**v).reshape(-1, 1), v )
	npars = 2**(v-1)

	nodes = np.zeros((v*npars,), dtype = int)
	pars = np.zeros((v*npars, v), dtype = int)

	for i in range(v):
		nodes[ (i*npars):(i+1)*npars ] = i
		pars[(i*npars):(i+1)*npars] = edgeconfs[ edgeconfs[:,i] == 0,: ]

	return nodes, pars

def apk(V, k):
	R = 0
	if V-k > 0:
		for v in range(1, V-k+1):
			R = R + ((2**k-1)**v) * (2**(k*(V-v-k))) * comb(V, k) * apk(V-k, v)
	else:
		R = 1
	return R

def ap(V):
	R = 0
	for v in range(1, V+1):
		R = R + apk(V, v)
	return int(R)

def iperm(p):
	plist = p.tolist()
	q = np.zeros( (len(p), ), dtype=int)
	for i in range(len(p)):
		ind = plist.index(i)
		q[i] = ind
	return q

def alldags(V):
	"""
	create a list of all possible dags with v nodes

	INPUT 
	v 			 number of variables v
	OUTPUT
	dags 		 list of (v, v) numpy array
	"""

	ndags = ap(V)
	Dindex = 0
	nallpairs = V*(V-1)//2 

	D = np.zeros( (ndags, V+nallpairs), dtype = int )
	B = np.zeros( (ndags, V*V), dtype = int)
	all_dags = [ np.zeros((V, V), dtype=int) ]

	Dindex = Dindex + 1
	D[0, 0:V] = np.arange(V)

	perm = permutations([ i for i in range(V) ])
	P = np.array(list(perm))
	perms = P.shape[0]

	# go through all connections
	for i in range(perms):
		for j in range(1, 2**nallpairs ):

			# convert decimal to binary
			binvec = util.dec_to_bin(j, nallpairs)

			# set connection matrix
			Bmat = np.zeros( (V,V), dtype = int )
			ridx, cidx = np.tril_indices( n=V, k=-1, m=V )
			Bmat[ridx, cidx] = binvec

			ip = iperm(P[i,:])
			Bmat = Bmat[:, ip]
			Bmat = Bmat[ip, :]

			Bvec = np.ndarray.flatten(Bmat, order='F')
			alreadyinB = False

			for jj in range(Dindex):
				if np.array_equal( Bvec, B[jj,:] ):
					alreadyinB = True
					break
			if alreadyinB:
				continue

			D[Dindex, :] = np.concatenate( (P[i,:], binvec) )
			B[Dindex, :] = Bvec
			all_dags.append(Bmat)
			Dindex += 1

	return all_dags