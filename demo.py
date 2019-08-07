import sys
import numpy as np

from BayesLiNGAM import BayesLiNGAM
sys.path.insert(1, './accessories') # add the path of the auxiliary functions
from gendata import gen_GCM_2vars, gen_GCM_3vars

def demo():
	X, b = gen_GCM_2vars(['2_1', 0, 0])
	# X, b = gen_GCM_3vars( ['3_1_1', 1, 0] )

	mdl = BayesLiNGAM(X, b)
	mdl.inference()
	if np.sum(np.isnan(mdl.prob)) == mdl.ndags:
		print('No valid DAG score! The results may be inaccurate!')
		return False

	print('\nThe true skeleton is')
	print(b)
	print('The estimated skeleton is')
	print(mdl.dags[ np.argmax(mdl.prob) ])
	return True

if __name__ == '__main__':
	demo()