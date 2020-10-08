import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from AccMeasure import acc_measure


def Estep(T, pi, mu, D, K, W):
	gamma = np.zeros((D,K))
	for i in range(0, D):
		numer_arr = np.zeros((K))
		for c in range(0, K):
			numer = 1
			pi_c = pi[c]
			for j in range(0, W):
				wp = np.power(mu[j, c], T[i, j])
				numer = numer * wp
			numer_arr[c] = numer * pi_c
		denom_sum = numer_arr.sum()
		gamma[i, :] = numer_arr / denom_sum
	return gamma


def Mstep(gamma, T, D, K, W):
	mu = np.empty((D,K))
	pi = np.empty((K))
	for j in range(0, W):
		for c in range(0, K):
			numer2 = 0
			denom2 = 0
			sum2 = 0
			for i in range(0, D):
				val = gamma[i, c] * T[i, j]
				numer2 = numer2 + val
				denom_arr = gamma[i,c] * T[i, :]
				denom2 = denom2 + np.sum(denom_arr)
			mu[j, c] = numer2 / denom2
		for c in range(0, K):
			pi[c] = np.sum(gamma[:, c]) / D
	return pi, mu


def Mstep_Mat(gamma, T, D, K, W):
	mu = np.empty((D,K))
	pi = np.empty((K))
	mu_bot = np.empty((D,K))
	mu_top = np.empty((D,K))

	j=0
	c=0
	for j in range(0, W):
		for c in range(0, K):
			mu_top[j,c] = np.sum(np.transpose(gamma[:,c]) * T[:, j])  #top
			mu_bot[j,c] = np.sum(gamma[:, c].reshape(gamma.shape[0],1) * T)

	mu = mu_top.copy() / mu_bot.copy()
	mu = np.nan_to_num(mu)

	for c in range(0, K):
		pi[c] = np.sum(gamma[:, c]) / D

	return pi, mu


def cluster(T, K, num_iters = 1000, epsilon = 1e-12, plot = False):
	"""

	:param bow:
		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
	:param K:
		number of topics
	:return:
		idx of size (num_doc), idx should be 1, 2, 3 or 4
	"""

	# preparations
	T = np.array(T)
	num_docs = T.shape[0]
	num_words = T.shape[1]

	# initialize output topic predictions (which is gamma)
	idx = np.zeros(num_docs)

	# Initialize expectations
	gamma = np.zeros((num_docs, K))
	# mu_arr = np.zeros((num_words, K))
	mu_arr = np.random.rand(num_words, K)

	# Same random value
	for j in range(0, K):
		mu_arr[:, j] = mu_arr[:, j] / np.sum(mu_arr[:, j])

	pi = np.empty((K))
	for k in range(0, K):
		pi[k] = np.random.uniform(0,1)
		# pi[k] = 1/K
	pi = pi / pi.sum()

	its = 0
	chg = float("inf")
	loss = 0

	conv_arr = []
	acc_arr = []

	while its <= num_iters and chg >= epsilon:
		gamma_last = gamma.copy()
		last_loss = loss

		# E-Step
		gamma = Estep(T, pi, mu_arr, num_docs, K, num_words)

		# M-Step
		# pi, mu_arr = Mstep(gamma, T, num_docs, K, num_words)
		pi, mu_arr = Mstep_Mat(gamma, T, num_docs, K, num_words)

		# Loss, Errors and Predictions
		loss = 0
		for i in range(0, num_words):
			loss_sub = 0
			for c in range(0, K):
				loss_sub = loss_sub + pi[c] * gamma[i, c]
			loss = loss + np.log(loss_sub)
		chg = np.linalg.norm(gamma - gamma_last)
		conv_arr.append(chg)
		idx = gamma.argmax(axis=1)

		its = its+1
		print("E-Step and M-Step complete on iteration {} with change of {}".format(its, chg))
		if plot:
			acc = acc_measure(idx)
			print('accuracy %.4f' % acc)
			acc_arr.append(acc)

	if plot:
		# Plotting
		plt.figure(1)
		plt.plot(range(1,its+1), acc_arr)
		plt.title("Accuracy Increase during Training")
		plt.xlabel("Iteration")
		plt.ylabel("Accuracy")
		plt.figure(2)
		plt.plot(range(1,its+1), conv_arr)
		plt.title("Convergence during Training")
		plt.xlabel("Iteration")
		plt.ylabel("Norm of Posterior Matrix Delta")
		plt.show()

	if idx.max() > 3 or idx.min() < 0 or idx.dtype != np.int64:
		raise ValueError("idx values must be 1,2,3, or 4!")
	return idx



def Estep_extra(T, pi, mu_jc, mu_ic, D, K, W):
	gamma = np.zeros((D,W,K))
	# for i in range(0, D):
	for c in range(0,K):
		print("Running Estep for cluster {} of {}".format(c, K))
		# numer_arr = np.zeros((K))
		# for c in range(0, K):
		pi_c = pi[c]
		for i in range(0, D):
			print("Running Estep for cluster {}, document {} of {}".format(c, i, D))
			numer = 1
			# pi_c = pi[c]
			for j in range(0, W):
				print("Running Estep for cluster {}, document {}, word {} of {}".format(c, i, j, W))
				wp_jc = np.power(mu_jc[j, c], T[i, j])
				wp_ic = np.power(mu_ic[i, c], T[i, j])
				numer = numer * wp_jc * wp_ic
				gamma[i,j,c] = numer * pi_c
			# numer_arr[c] = numer * pi_c
		# denom_sum = numer_arr.sum()
		# gamma[i, ] = numer_arr / denom_sum
		gamma[:,:,c] = gamma[:,:,c] / np.sum(gamma[:,:,c] )
	return gamma


def Mstep_extra(gamma, T, D, K, W):
	mu_ic = np.empty((D, K))
	mu_jc = np.empty((W, K))

	pi = np.empty((K))
	mu_ic_bot = np.empty((D, K))
	mu_ic_top = np.empty((D, K))

	mu_jc_bot = np.empty((W, K))
	mu_jc_top = np.empty((W, K))

	print("Running Mstep to calculate mu_jc")
	j = 0
	c = 0
	for j in range(0, W):
		for c in range(0, K):
			mu_jc_top[j, c] = np.sum(np.transpose(gamma[:,j, c]) * T[:, j])  # top
			mu_jc_bot[j, c] = np.sum(gamma[:,j, c].reshape(1,gamma.shape[0]) * T)

	mu_jc = mu_jc_top.copy() / mu_jc_bot.copy()
	mu_jc = np.nan_to_num(mu_jc)

	print("Running Mstep to calculate mu_ic")
	i = 0
	c = 0
	for i in range(0, D):
		for c in range(0, K):
			mu_ic_top[i, c] = np.sum(np.transpose(gamma[i,:, c]) * np.transpose(T[i, :]))  # top
			mu_ic_bot[i, c] = np.sum(gamma[i,:, c].reshape(1,gamma.shape[1]) * np.transpose(T))

	mu_ic = mu_ic_top.copy() / mu_ic_bot.copy()
	mu_ic = np.nan_to_num(mu_ic)

	print("Running MStep to calculate pi")
	for c in range(0, K):
		pi[c] = np.sum(gamma[:, :, c]) / (D*W)

	return pi, mu_jc, mu_ic


def cluster_extra(T, K, num_iters = 1000, epsilon = 1e-12, plot = False):
	# preparations
	# T = np.array(T)
	num_docs = T.shape[0]
	num_words = T.shape[1]

	# Initialize expectations
	gamma = np.random.uniform(0,1,(num_docs, num_words, K))
	mu_jc = np.random.rand(num_words, K)
	mu_ic = np.random.rand(num_docs, K)

	# Same random value
	for j in range(0, K):
		mu_jc[:, j] = mu_jc[:, j] / np.sum(mu_jc[:, j])
		mu_ic[:, j] = mu_ic[:, j] / np.sum(mu_ic[:, j])

	pi = np.empty((K))
	for k in range(0, K):
		pi[k] = np.random.uniform(0,1)
		# pi[k] = 1/K
	pi = pi / pi.sum()

	its = 0
	chg = float("inf")
	loss = 0

	conv_arr = []
	acc_arr = []

	while its <= num_iters and chg >= epsilon:
		gamma_last = gamma.copy()

		# E-step
		print("Running Estep!")
		gamma = Estep_extra(T, pi, mu_jc, mu_ic, num_docs, K, num_words)



		chg = np.linalg.norm(gamma - gamma_last)
		conv_arr.append(chg)
		# idx = gamma.argmax(axis=1)

		# M-step
		print("Running Mstep!")
		pi, mu_jc, mu_ic = Mstep_extra(gamma, T, num_docs, K, num_words)

		its = its+1
		print("E-Step and M-Step complete on iteration {} with change of {}".format(its, chg))
	return mu_jc