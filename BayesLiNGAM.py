"""
Implementation of bayeslingam proposed in [1]
(Note: only the GL parametrization, i.e., Eq.(5),(6) in [1])

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-08-05

[1] Hoyer, Patrik O., and Antti Hyttinen. "Bayesian discovery of linear acyclic causal models." 
    In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence,
    pp. 240-248. AUAI Press, 2009.
"""
from __future__ import division
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.optimize import minimize
from copy import deepcopy

import sys
sys.path.insert(1, './accessories') # add the path of the auxiliary functions
from dags import allfamilies, alldags
import util

class BayesLiNGAM(object):
	def __init__(self, data, B=None, verbose=0):
		shifter = StandardScaler().fit(data)
		self.X = shifter.transform(data) # the *standardized* input data
		self.N, self.V = self.X.shape # N: the number of instances, V: the number of variables

		self.dags = alldags(self.V) # list of all possible dags
		self.ndags = len(self.dags) # the number of all dags

		# prior of the parameters in GL approximation
		self.priorGL = { 'alp': [0.0, 1.0], # [mu, var]
				'bet': [0.0, 1.0],
				'b': [0.0, 1.0] }

		if B is not None: # when the true dag is given, store its index in self.gtidx
			B_sklt = deepcopy(B)
			B_sklt[ B_sklt != 0 ] = 1
			for d in range(self.ndags):
				if np.array_equal(B_sklt, self.dags[d]):
					self.gtidx = d
					break
		if verbose == 1:
			print('Initialization done.')

	def logpriorGL(self, p):
		"""
		Compute the log of priors of parameters alpha, beta and weights b

		INPUT
		  p 		 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array

		OUTPUT
		  logpr 	 log of priors of all parameters, float
		"""

		# parameter alpha
		logpr_alp = - 0.5*np.log(2*np.pi) - 0.5*np.log( self.priorGL['alp'][1] ) - 0.5*(p[0] - self.priorGL['alp'][0])**2 / self.priorGL['alp'][1]
		# parameter beta
		logpr_bet = - 0.5*np.log(2*np.pi) - 0.5*np.log( self.priorGL['bet'][1] ) - 0.5*(p[1] - self.priorGL['bet'][0])**2 / self.priorGL['bet'][1]
		# weights b
		L_b = len(p) - 2
		if L_b > 0:
			b = p[2:]
			logpr_b = (-L_b/2)*np.log(2*np.pi) - (L_b/2)*np.log( self.priorGL['b'][1] ) - 0.5*np.sum( (b - self.priorGL['b'][0])**2 )/ self.priorGL['b'][1]
		else:
			logpr_b = 0

		return logpr_alp + logpr_bet + logpr_b

	def posterior(self, params, obs):
		"""
		Compute the negative log of posterior probability

		INPUT
		  params 		 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array
		  obs 			 instances of current variable and its parents, [var, pars]

		OUTPUT
		  logpost 		 negative log of posterior probability, float
		"""

		return - self.logpriorGL(params) - loglikelihood(params, obs)

	def numerical_coefs(self, params, obs):
		"""
		Compute the first and second-order differentiation w.r.t. params numerically

		INPUT
		  params 		 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array
		  obs 			 instances of current variable and its parents, [var, pars]

		OUTPUT
		  reg.coef_ 	 the first and second order of differentiation of params, numpy array
		"""

		k = len(params)
		step = 1e-3

		P = np.zeros((3**k, k), dtype = int)
		for i in range(3**k):
			P[i,:] = util.dec_to_trin(i,k)
		P = P-1
		P = P*step

		Y = np.zeros((3**k,), dtype = float)
		for i in range(3**k):
			Y[i] = self.posterior(params+P[i,:], obs)

		Q = util.colProduct(P)
		reg = LinearRegression(fit_intercept=False).fit( np.concatenate( (P, Q), axis=1 ), Y)
		return reg.coef_

	def beginningvaluesGL(self, obs, step = 0.2):
		"""
		Compute the initial value of params

		INPUT
		  obs 		 instances of current variable and its parents, [var, pars]

		OUTPUT
		  p 		 the initial value of params, (2+self.V, ) numpy array
		"""
		
		y = obs[0] # instances of child node in the current family
		X = obs[1] # instances of parent nodes in the current family
		cur_N, cur_V = X.shape
		p = np.zeros((2+cur_V,), dtype = float) # [alp, bet, b_1, ..., b_v]

		if cur_V >= 1:
			reg = LinearRegression(fit_intercept=False).fit(X=X, y=y)
			p[2:] = reg.coef_
			r = y - reg.predict(X)
		else:
			r = -y

		sr = np.sum(abs(r))
		sr2 = np.sum(r**2)

		p1 = np.arange(-10, 10.1, step)
		p2 = np.arange(-10, 10.1, step)
		L_p = len(p1)
		Z = np.matrix(np.ones((L_p,L_p)) * np.NINF)
		for l1 in range(L_p):
			for l2 in range(L_p):
				p[0] = p1[l1]
				p[1] = p2[l2]
				logpost = loglikelihood2sufficient(p, sr, sr2, cur_N) + self.logpriorGL( p )

				if np.isfinite(logpost):
					Z[l1, l2] = logpost

		idx_max = np.unravel_index(np.argmax(Z), (L_p,L_p))
		p[0], p[1] = p1[idx_max[0]], p2[idx_max[1]]
		return p

	def posterior_grad_ori(self, params, obs):
		"""
		Compute the gradients of params

		INPUT
		  params 	 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array
		  obs 		 instances of current variable and its parents, [var, pars]

		OUTPUT
		  G 		 the gradients of params, (2+self.V, ) numpy array
		"""
		k = len(params)
		G = self.numerical_coefs(params, obs)[0:k]
		return G

	def posterior_hessian_ori(self, params, obs):
		"""
		Compute the Hessian of params

		INPUT
		  params 	 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array
		  obs 		 instances of current variable and its parents, [var, pars]

		OUTPUT
		  H 		 the Hessian of params, (2+self.V, 2+self.V) numpy array
		"""
		k = len(params)
		coefs = self.numerical_coefs(params, obs)
		h = coefs[k:]

		H = np.zeros( (k,k), dtype = float )
		cnt = 0
		for col in range(k):
			for row in range(col, k):
				H[row, col ] = h[cnt]
				cnt += 1
		H = H + H.T
		return H

	def scores_families(self, cu_X):
		"""
		Compute the integral value ( Eq.(2) in [1] ) of all families

		INPUT
		  cu_X 			 instances in the current family
		OUTPUT
		  families 		 every family and its integral value, dictionary
		"""
		nodes, pars = allfamilies(self.V) # compute the list of all possible families
		num_families = len(nodes) # total number of families

		scores = np.zeros( (num_families, ), dtype=float )
		for f in range( num_families ):
			nonzero_idx = np.nonzero( pars[f,:] )[0]
			cu_obs = [ cu_X[ :, nodes[f] ], cu_X[ :, nonzero_idx ] ]

			p0 = self.beginningvaluesGL(cu_obs)
			fval = self.posterior(p0, cu_obs)

			min_obj = minimize(self.posterior, p0, cu_obs, method='BFGS', jac=self.posterior_grad_ori)
			min_p = min_obj.x
			H = self.posterior_hessian_ori(min_p, cu_obs)

			if min_obj.success == True:
				mi = - self.posterior(min_p, cu_obs)
				di = np.log( np.linalg.det( H ) )
				if not np.isfinite(di):
					mi = float('-inf')
					di = 0
			else:
				eig_vals, eig_vecs = np.linalg.eig(H)
				eig_vals.sort()
				if eig_vals[0] > 0:
					mi = self.posterior(min_p, cu_obs)
					di = np.log( np.linalg.det( H ) )
				else:
					mi = float('-inf')
					di = 0

			scores[f] = mi - 0.5*di + 0.5*len(min_p)*np.log(np.pi)

		self.families = {'node': nodes,
					'par': pars,
					'score': scores}

	def score_dag(self, dag):
		"""
		Compute the score of each dag using the scores of all families

		INPUT
		  dag 			 current dag, (self.V, self.V) numpy array
		OUTPUT
		  cur_score 	 the score of the input dag
		"""
		cur_score = 0
		num_families = len( self.families['node'] )

		for i in range(self.V):
			for f in range(num_families):
				if self.families['node'][f] == i and np.array_equal( dag[i,:], self.families['par'][f,:] ):
					cur_score += self.families['score'][f]
					break
		return cur_score

	def inference(self):

		# compute the score of all possible families (a family is a node a parent set)
		self.scores_families(self.X)

		# compute the score of each dag
		self.loglike = np.zeros((self.ndags,), dtype=float)
		for i in range(self.ndags):
			self.loglike[i] = self.score_dag(self.dags[i])

		# compute the possibily of each dag
		self.prob = np.exp( self.loglike - max(self.loglike) )
		self.prob = self.prob / np.sum(self.prob)
		print('The estimated probabilities of all DAGS are:')
		print(self.prob)

def loglikelihood2sufficient(params, sr, sr2, N):
	"""
	Compute the log-likelihood using sufficient statistics of the noise

	INPUT
	  params 	 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array
	  sr 		 first order sufficient statistics of the noise
	  sr2 		 second order sufficient statistics of the noise
	  N 		 the number of instances
	OUTPUT
	   			the log-likelihood
	"""
	alp, bet = params[0], np.exp(params[1])

	erf_term = norm.cdf(0, alp/(2*bet), 1/(np.sqrt(2*bet)))
	if erf_term == 0:
		return float('-inf')
	else:
		erf_term = np.log( erf_term )

	# erf_term = np.log( norm.cdf(0, alp/(2*bet), 1/(np.sqrt(2*bet))) )
	logZ = N*( np.log(2) + alp**2/(4*bet) + 0.5*np.log(np.pi) - 0.5*np.log(bet) + erf_term)
	Numerator = (-alp)*sr - bet*sr2

	return Numerator - logZ

def residual(b, obs):
	"""
	compute the residuals of the current family

	INPUT
	  b 	 current weights [b_1, ..., b_v], (self.V, ) numpy array
	  obs 		 instances of current variable and its parents, [var, pars]
	OUTPUT
	   			the corresponding residuals
	"""

	y = obs[0]
	X = obs[1]
	V = X.shape[1]

	if V == 0:
		r = -y
	else:
		r = y - np.dot( X, b )
	return r

def loglikelihood(params, obs):
	"""
	Compute the log-likelihood of instances in obs

	INPUT
	  params 	 current parameters [alpha, log(beta), b_1, ..., b_v], (k, ) numpy array
	  obs 		 instances of current variable and its parents, [var, pars]
	OUTPUT
	   			the log-likelihood
	"""
	approximative=False
	factor=10
	y = obs[0]
	X = obs[1]

	alp = params[0]
	bet = np.exp( params[1] )

	r = residual( params[2:], obs)
	if approximative:
		sr = np.sum( np.log( np.cosh(factor*r) )/factor )
	else:
		sr = np.sum( abs(r) )
		sr2 = np.sum(r**2)

	erf_term = norm.cdf(0, alp/(2*bet), 1/(np.sqrt(2*bet)))
	if erf_term == 0:
		return float('-inf')
	else:
		erf_term = np.log( erf_term )
	logZ = len(y)*( np.log(2) + alp**2/(4*bet) + 0.5*np.log(np.pi) - 0.5*np.log(bet) + erf_term )
	Numerator = -alp*sr - bet*sr2
	return Numerator - logZ