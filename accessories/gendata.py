from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import numpy as np

def gen_data_given_model(b, s, c, n_samples=2000, random_state=0):
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
	xs, b_, c_ : Tuple
		`xs` is observation matrix, where `xs.shape==(n_samples, n_features)`. 
		`b_` is permuted coefficient matrix. Note that rows of `b_` correspond
		to columns of `xs`. `c_` if permuted mean vectors. 

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
		xs[:, i] = ss[:, i] + ss.dot(b[i, :]) + c[i]
		# print xs

	# print ss
	# print xs
	# Permute variables
	# p = rng.permutation(n_vars)
	# xs[:, :] = xs[:, p]
	# b_ = deepcopy(b)
	# c_ = deepcopy(c)
	# b_[:, :] = b_[p, :]
	# b_[:, :] = b_[:, p]
	# c_[:] = c[p]

	# return xs, b_, c_

	return xs

def gen_GCM_2vars(flags):

	# two variables
	# --- no edge
	if flags[0] == '2_0':
		b = np.array([[0.0, 0.0], [0.0, 0.0]])
	# --- one edge
	elif flags[0] == '2_1_1':
		b = np.array([[0.0, 1.0], [0.0, 0.0]])
	elif flags[0] == '2_1_2':
		b = np.array([[0.0, 0.0], [1.0, 0.0]])
	else:
		b = np.array([[0.0, 0.0], [0.0, 0.0]])

	if flags[1] == 1:
		s = np.array([0.25, 0.25])
	elif flags[1] == 2:
		s = np.array([0.5, 0.5])
	elif flags[1] == 3:
		s = np.array([1.0, 1.0])
	elif flags[1] == 4:
		s = np.array([1.5, 1.5])
	else:
		s = np.array([1.0, 1.0])

	if flags[2] == 1:
		c = np.array([5.0, 6.0])
	else:
		c = np.array([0.0, 0.0])

	# xs, b_, c_ = gen_data_given_model(b, s, c)
	xs = gen_data_given_model(b, s, c)

	return xs, b


def gen_GCM_3vars(flags):

	# three variables
	# --- no edge
	if flags[0] == '3_0':
		b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
	# --- one edge
	elif flags[0] == '3_1_1':
		b = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
	elif flags[0] == '3_1_2':
		b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
	elif flags[0] == '3_1_3':
		b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
	# --- two edges
	elif flags[0] == '3_2_1':
		b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
	elif flags[0] == '3_2_2':
		b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
	elif flags[0] == '3_2_3':
		b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
	# --- three edges
	elif flags[0] == '3_3_1':
		b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
	else:
		b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

	if flags[1] == 1:
		s = np.array([0.25, 0.25, 0.25])
	elif flags[1] == 2:
		s = np.array([0.5, 0.5, 0.5])
	elif flags[1] == 3:
		s = np.array([1.0, 1.0, 1.0])
	elif flags[1] == 4:
		s = np.array([1.5, 1.5, 1.5])
	else:
		s = np.array([1.0, 1.0, 1.0])

	if flags[2] == 1:
		c = np.array([5.0, 6.0, 7.0])
	else:
		c = np.array([0.0, 0.0, 0.0])

	# xs, b_, c_ = gen_data_given_model(b, s, c)
	xs = gen_data_given_model(b, s, c)

	return xs, b

def gen_syn_2(s1, s2):
	X1, b1 = gen_GCM([s1, 1, 1])
	scaler1 = StandardScaler(with_std=False)
	scaler1.fit(X1)
	data1 = scaler1.transform(X1)

	X2, b2 = gen_GCM([s2, 3, 1])
	scaler2 = StandardScaler(with_std=False)
	scaler2.fit(X2)
	data2 = scaler2.transform(X2)

	data = np.concatenate( (data1, data2) )
	label_true = np.array( [1]*X1.shape[0] + [2]*X2.shape[0]  )
	# print label_true.shape
	dag1 = deepcopy(b1)
	dag2 = deepcopy(b2)

	dag1[ dag1 != 0 ] = 1
	dag2[ dag2 != 0 ] = 1

	return data, [dag1, dag2], label_true

def gen_syn_3(s1, s2, s3):
	X1, b1 = gen_GCM([s1, 3, 1])
	scaler1 = StandardScaler(with_std=False)
	scaler1.fit(X1)
	data1 = scaler1.transform(X1)

	X2, b2 = gen_GCM([s2, 2, 1])
	scaler2 = StandardScaler(with_std=False)
	scaler2.fit(X2)
	data2 = scaler2.transform(X2)

	X3, b3 = gen_GCM([s3, 4, 1])
	scaler3 = StandardScaler(with_std=False)
	scaler3.fit(X3)
	data3 = scaler3.transform(X3)

	data = np.concatenate( (data1, data2, data3) )
	label_true = np.array( [1]*X1.shape[0] + [2]*X2.shape[0] + [3]*X3.shape[0] )
	# print label_true.shape
	dag1 = deepcopy(b1)
	dag2 = deepcopy(b2)
	dag3 = deepcopy(b3)

	dag1[ dag1 != 0 ] = 1
	dag2[ dag2 != 0 ] = 1
	dag3[ dag3 != 0 ] = 1

	return data, [dag1, dag2, dag3], label_true
