import sys
import numpy as np

from BayesLiNGAM import BayesLiNGAM
sys.path.insert(1, './accessories') # add the path of the auxiliary functions
from gendata import gen_GCM

def demo():
	X, B = gen_GCM(2, 1)

	mdl = BayesLiNGAM(X, B)
	mdl.inference()
	if np.sum(np.isnan(mdl.prob)) == mdl.ndags:
		print('No valid DAG score! The results may be inaccurate!')
		return False

	print('\nThe true skeleton is')
	print(B)
	print('The estimated skeleton is')
	print(mdl.dags[ np.argmax(mdl.prob) ])
	return True

if __name__ == '__main__':
	demo()