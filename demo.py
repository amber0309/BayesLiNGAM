import numpy as np
from BayesLiNGAM import BayesLiNGAM
from accessories.gendata import gen_GCM

def demo():
	X, B = gen_GCM(2, 1) # generate synthetic data, 2 variables, 1 edge

	mdl = BayesLiNGAM(X, B)
	mdl.inference()

	print('The true graph structure is:')
	print(B)

if __name__ == '__main__':
	demo()