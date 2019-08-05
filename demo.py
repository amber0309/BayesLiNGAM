import sys
import numpy as np

from BayesLiNGAM import BayesLiNGAM
sys.path.insert(1, './accessories') # add the path of the auxiliary functions
from gendata import gen_GCM_2vars, gen_GCM_3vars

def demo():
	X, b = gen_GCM_2vars(['2_1_1', 0, 0])
	# X, b = gen_GCM_3vars( ['3_1_1', 1, 0] )

	mdl = BayesLiNGAM(X, b)
	mdl.inference()
	if np.sum(np.isnan(mdl.prob)) == mdl.ndags:
		print('No valid DAG score! The results may be inaccurate!')
	else:
		if np.argmax(mdl.prob) == mdl.gtidx:
			print('\nThe true DAG is')
			print(b)
			print('The estimated DAG is')
			print(mdl.dags[ np.argmax(mdl.prob) ])

if __name__ == '__main__':
	demo()