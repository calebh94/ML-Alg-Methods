import numpy as np
from numpy import random
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


def Estep_Mat(T, pi, mu, D, K, W):
	gamma = np.zeros((D,K))
	gamma_bot = np.zeros((D,K))
	gamma_top = np.zeros((D,K))
	i=1
	c=2
	for i in range(0, W):
		for c in range(0, K):
			gamma_top[i,c] = pi[c] * np.prod(np.power(mu[:, c], T[i, :]))
		# for c in range(0, K):
		gamma_bot[i,:] = np.sum(gamma_top[i,:])
	# gamma = gamma_top / gamma_bot
	gamma = np.divide(gamma_top, gamma_bot)
	# gamma = np.nan_to_num(gamma)
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
				denom_arr = gamma[i,c] * T[i, :]  # TODO: matrix math applied elswhere?
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


def cluster(T, K, num_iters = 1000, epsilon = 1e-12):
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

	# Making assumptions on documents and clusters for the words
	# split = num_docs / K
	# for p in range(0, K):
	#
	# 	word_cnts = np.sum(T[int(p*split):int(p*split+split), :], axis=0)
	# 	word_cnts = word_cnts / word_cnts.sum()
	# 	mu_arr[:, p] = word_cnts

	pi = np.empty((K))
	for k in range(0, K):
		pi[k] = np.random.uniform(0,1)
		# pi[k] = 1/K
	pi = pi / pi.sum()

	# print(mu_arr)
	# print(pi)

	its = 0
	chg = float("inf")
	loss = 0

	conv_arr = []
	acc_arr = []

	while its <= num_iters and chg >= epsilon:
		gamma_last = gamma.copy()
		last_loss = loss
		# E-Step, TODO: move to separate function call
		gamma = Estep(T, pi, mu_arr, num_docs, K, num_words)
		# for i in range(0, num_docs):
		# 	numer_arr = np.empty((K))
		# 	for c in range(0, K):
		# 		numer = 1
		# 		pi_c = pi[c]
		# 		for j in range(0, num_words):
		# 			wp = np.power(mu_arr[j, c], T[i, j])
		# 			numer = numer * wp
		# 		numer_arr[c] = numer * pi_c
		# 	denom_sum = numer_arr.sum()
		# 	gamma[i, :] = numer_arr / denom_sum

		# Calculate Loss
		loss = 0
		for i in range(0, num_words):
			loss_sub = 0
			for c in range(0, K):
				loss_sub = loss_sub + pi[c] * gamma[i, c]
			loss = loss + np.log(loss_sub)

		# err_arr = np.abs(gamma_last - gamma)
		# err = np.sum(err_arr)
		# chg = np.abs(last_loss - loss)
		chg = np.linalg.norm(gamma - gamma_last)
		conv_arr.append(chg)

		idx = gamma.argmax(axis=1)

		# M-Step, TODO: move to seperate function call
		pi, mu_arr = Mstep(gamma, T, num_docs, K, num_words)
		# mu_arr_last = mu_arr
		# pi_last = pi
		# for j in range(0, num_words):
		#
		# 	for c in range(0, K):
		#
		# 		numer2 = 0
		# 		denom2 = 0
		# 		sum2 = 0
		# 		for i in range(0, num_docs):
		# 			val = gamma[i, c] * T[i, j]
		# 			numer2 = numer2 + val
		#
		# 			denom_arr = gamma[i,c] * T[i, :]  # TODO: matrix math applied elswhere?
		# 			denom2 = denom2 + np.sum(denom_arr)
		#
		#
		# 		mu_arr[j, c] = numer2 / denom2
		#
		# for c in range(0, K):
		# 	pi[c] = np.sum(gamma[:, c]) / num_docs

		its = its+1
		print("E-Step and M-Step complete on iteration {} with change of {}".format(its, chg))
		acc = acc_measure(idx)
		print('accuracy %.4f' % (acc))
		acc_arr.append(acc)


	if idx.max() > 3 or idx.min() < 0 or idx.dtype != np.int64:
		raise ValueError("idx values must be 1,2,3, or 4!")
	return idx
