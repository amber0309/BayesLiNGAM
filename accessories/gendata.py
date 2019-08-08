from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import numpy as np

def gen_data_given_model(b, s, c, n_samples=1000, random_state=4):
	"""Generate artificial data based on the given model.

	Parameters
	----------
	b : numpy.ndarray, shape=(n_features, n_features)
		Strictly lower triangular coefficient matrix. 
		NOTE: Each row of `b` corresponds to each variable, i.e., X = BX. 
	s : numpy.ndarray, shape=(n_features,)
		Scales of disturbance variables.
	c : numpy.ndarray, shape=(n_features,)
		Means of observed variables. 

	Returns
	-------
	xs, b_, : Tuple
		`xs` is observation matrix, where `xs.shape==(n_samples, n_features)`. 
		`b_` is permuted coefficient matrix. Note that rows of `b_` correspond
		to columns of `xs`.

	"""
	rng = np.random.RandomState(random_state)
	n_vars = b.shape[0]

	# Check args
	assert(b.shape == (n_vars, n_vars))
	assert(s.shape == (n_vars,))
	assert(np.sum(np.abs(np.diag(b))) == 0)
	np.allclose(b, np.tril(b))

	# Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
	# (<1 gives subgaussian, >1 gives supergaussian)
	# q = rng.rand(n_vars) * 1.1 + 0.5    
	# ixs = np.where(q > 0.8)
	# q[ixs] = q[ixs] + 0.4

	# Generates disturbance variables
	ss = rng.randn(n_samples, n_vars)
	# ss = np.sign(ss) * (np.abs(ss)**q)

	# Normalizes the disturbance variables to have the appropriate scales
	ss = ss / np.std(ss, axis=0) * s
	# Generate the data one component at a time
	xs = np.zeros((n_samples, n_vars))
	for i in range(n_vars):
		# NOTE: columns of xs and ss correspond to rows of b
		xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

	# Permute variables
	p = rng.permutation(n_vars)
	xs[:, :] = xs[:, p]
	b_ = deepcopy(b)
	c_ = deepcopy(c)
	b_[:, :] = b_[p, :]
	b_[:, :] = b_[:, p]
	c_[:] = c[p]

	return xs, b_

def gen_GCM(n_vars, n_edges, stds=0, biases=0):

	B = np.zeros( (n_vars, n_vars), dtype=float )
	max_nedge = n_vars*(n_vars-1)/2

	trilB_vec = np.array([0] * (max_nedge - n_edges) + [1] * n_edges)
	np.random.shuffle( trilB_vec )
	ridx, cidx = np.tril_indices( n=n_vars, k=-1, m=n_vars )
	B[ridx, cidx] = trilB_vec

	stds_vec = np.ones( (n_vars,), dtype=float ) * stds
	bias_vec = np.ones( (n_vars,), dtype=float ) * biases

	xs, b_ = gen_data_given_model(B, stds_vec, bias_vec)

	return xs, b_