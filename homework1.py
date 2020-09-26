import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os
import time


'''
K-Means Implementation and Functions
---------------------------------------
'''
def distance3d(pixel, cluster):
	pixel = pixel.astype(float)
	cluster = cluster.astype(float)
	return (pixel[0] - cluster[0])**2 + (pixel[1] - cluster[1])**2 + (pixel[2] - cluster[2])**2


def assignclusters(pixels, clusters):
	assignments = np.zeros((pixels.shape[0], 1), np.int)
	for i in range(0, pixels.shape[0]):
		# assign pixel to cluster
		pixel = pixels[i][:]
		assign = 0
		min_dis = float("inf")
		for j in range(0, clusters.shape[0]):
			cluster = clusters[j][:]
			distance = distance3d(pixel, cluster)
			if distance < min_dis:
				min_dis = distance
				assign = j
		assignments[i] = assign
	groups = np.linspace(0, clusters.shape[0] - 1, num=clusters.shape[0], dtype=np.int)
	mask = np.isin(groups, np.unique(assignments))
	# Check for empty clusters and randomly assign to keep algorithm running (for large # of clusters)
	if np.isin(False,mask):
		print("Some empty clusters found, random sampling new points for cluster")
		for i in range(0, mask.shape[0]):
			if mask[i] == False:
				selections = np.random.randint(0, assignments.shape[0], 1)
				assignments[selections] = i
	return assignments


def centeradjustment(assignments, pixels, K):
	pixel_locations = np.concatenate((assignments, pixels), axis=1)
	pixel_locations.sort(axis=0)
	cluster_sum = np.zeros((K,3), np.int)
	cluster_num = 0
	cluster_count = 0
	for i in range(0, pixel_locations.shape[0]):
		if pixel_locations[i][0] > cluster_count:
			cluster_sum[cluster_count] = cluster_sum[cluster_count] / cluster_num
			cluster_count = cluster_count + 1
			cluster_num = 0
		cluster_num = cluster_num + 1
		cluster_sum[cluster_count] = cluster_sum[cluster_count] + pixel_locations[i][1:4]
	cluster_sum[-1] = cluster_sum[-1] / cluster_num  # last group
	clusters = cluster_sum
	return clusters


def mykmeans(pixels, K):
	centroids_ind = np.random.randint(0, pixels.shape[0], (K, 1))  # initialize as random index location
	centroids = pixels[centroids_ind.squeeze()]
	steps = 100
	cnt = 1
	epsilon = 1e-1
	converged = False
	print("Clustering iterations for K-Means are beginning...")
	while not converged:
		# CLUSTER ASSIGNMENT
		classes = assignclusters(pixels, centroids)
		centroids_last = centroids
		# CLUSTER ADJUSTMENT
		centroids = centeradjustment(classes, pixels, K)
		if (cnt % 10 == 0 or cnt == 1):
			print("K-means Iteration #{}".format(cnt))
			print("Centroid clusters: \n {}".format(centroids))
		cnt = cnt + 1
		if np.linalg.norm(centroids-centroids_last, axis=0).sum() < epsilon:
			converged = True
			print("K-Means Converged after {} iterations".format(cnt))
		elif cnt >= steps:
			converged = True
			print("K-Means Stopped after reaching max {} iterations".format(cnt))
		else:
			pass
	return classes, centroids


'''
K-Mediod Implementation and Functions
---------------------------------------
'''
def distance_function(pixel, cluster, option):
	pixel = pixel.astype(float)
	cluster = cluster.astype(float)
	if option == 0: # euclidean
		distance = (pixel[0] - cluster[0]) ** 2 + (pixel[1] - cluster[1]) ** 2 + (pixel[2] - cluster[2]) ** 2
	elif option == 1:  # L1
		distance = abs(pixel[0] - cluster[0]) + abs(pixel[1] - cluster[1]) + abs(pixel[2] - cluster[2])
	else:
		distance = 0
	return distance


def kmed_assignclusters(pixels, clusters, K, option):
	assignments_r = np.zeros((pixels.shape[0], K))
	# option = 0
	for i in range(0, pixels.shape[0]):
		# assign pixel to cluster
		pixel = pixels[i][:]
		assign = 0
		min_dis = float("inf")
		for j in range(0, K):
			cluster = clusters[j][:]
			# distance = distance3d(pixel, cluster) #TODO: change distance function
			distance = distance_function(pixel, cluster, option)
			if distance < min_dis:
				min_dis = distance
				assign = j
		assignments_r[i][assign] = 1

	return assignments_r


def distance_matrix(pixels):
	print("Calculating Distance Matrix...")
	P = pixels-np.mean(pixels, axis=0)
	Pt = np.transpose(P)
	q=np.linalg.norm(P, axis=1)
	D = q + np.transpose(q) - 2*np.matmul(P,Pt)
	print("Distance matrix calculated!")
	return D


def distance_matrix_manhattan(pixels):
	print("Calculating Manhattan Distance Matrix...")
	P = pixels-np.mean(pixels, axis=0)
	Dj = np.zeros((P.shape[0], P.shape[0], 3))
	q = np.linalg.norm(P,axis=1)
	for j in range(0,3):
		Pj = P[:, j]
		Pjt = np.transpose(Pj)
		q = np.square(Pj)
		qt = np.transpose(q)
		Dj[:,:,j] = q + qt - 2*np.matmul(Pj, Pjt)

	D = np.sum(Dj, axis=2)
	print("Distance Manhattan matrix calculated!")

	return D


def distance_matrix_chebychev(pixels):
	print("Calculating Chebychev Distance Matrix...")
	P = pixels-np.mean(pixels, axis=0)
	Dj = np.zeros((P.shape[0], P.shape[0], 3))
	q = np.linalg.norm(P,axis=1)
	for j in range(0,3):
		Pj = P[:, j]
		Pjt = np.transpose(Pj)
		q = np.square(Pj)
		qt = np.transpose(q)
		Dj[:,:,j] = q + qt - 2*np.matmul(Pj, Pjt)

	D = np.max(Dj, axis=2)
	print("Distance Chebychev matrix calculated!")
	return D


def kmed_assignclusters_matrix(pixels, centroids_ind, K, D):
	assignments_r = np.zeros((pixels.shape[0], K))
	for i in range(0, pixels.shape[0]):
		# assign pixel to cluster
		pixel = pixels[i][:]
		assign = 0
		min_dis = float("inf")
		for j in range(0, K):
			ind = centroids_ind[j][:]
			distance = D[i, j]
			# distance = # find in distance matrix
			if distance < min_dis:
				min_dis = distance
				assign = j
		assignments_r[i][assign] = 1

	# groups = np.linspace(0, K - 1, num=K, dtype=np.int)
	# mask = np.isin(groups, np.unique(assignments_r))
	# # Check for empty clusters and randomly assign to keep algorithm running (for large # of clusters)
	# if np.isin(False, mask):
	# 	print("Some empty clusters found, random sampling new points for cluster")
	# 	for i in range(0, mask.shape[0]):
	# 		if mask[i] == False:
	# 			selections = np.random.randint(0, assignments_r.shape[0], 1)
	# 			assignments_r[selections] = i

	return assignments_r


def kmed_findcentroid_matrix(cluster_pixels, pixels, D):
	best_centroid = 0
	best_distancesum = float("inf")
	distancesum = 0.0
	for i in range(0,cluster_pixels.shape[0]):
		# print("{} of {} data points checked for dissimilarity in this cluster".format(i, cluster_pixels.shape[0]))
		pixel_sel = pixels[cluster_pixels[i]]
		distancesum = 0.0
		for j in range(0, cluster_pixels.shape[0]):
			if i==j:
				continue
			else:
				distance = D[i,j]
				distancesum = distancesum + distance
		if distancesum < best_distancesum:
			best_distancesum = distancesum
			best_centroid = i

	return best_centroid


def kmed_findcentroid(cluster_pixels, pixels):
	best_centroid = 0
	best_distancesum = float("inf")
	distancesum = 0.0
	for i in range(0,cluster_pixels.shape[0]):
		print("{} of {} data points checked for dissimilarity in this cluster".format(i, cluster_pixels.shape[0]))
		pixel_sel = pixels[cluster_pixels[i]]
		distancesum = 0.0
		for j in range(0, cluster_pixels.shape[0]):
			if i==j:
				continue
			else:
				distance = distance_function(pixel_sel.squeeze(), pixels[cluster_pixels[j]].squeeze(), option=1)
				distancesum = distancesum + distance
		if distancesum < best_distancesum:
			best_distancesum = distancesum
			best_centroid = i

	return best_centroid


def kmed_L2centroid(cluster_pixels, pixels):
	best_centroid = 0
	best_distance = float("inf")
	mean = pixels[cluster_pixels].mean(axis=0)
	for i in range(0, cluster_pixels.shape[0]):
		# print("{} of {} data points checked for dissimilarity in this cluster".format(i, cluster_pixels.shape[0]))
		pixel_sel = pixels[cluster_pixels[i]]
		distance = distance_function(pixel_sel.squeeze(), mean.squeeze(), option=0)
		if distance < best_distance:
			best_distance = distance
			best_centroid = cluster_pixels[i].squeeze()
	return best_centroid


def kmed_updateclusters(assign_r, pixels, K):
	cluster_inds = np.zeros((K,1), np.int)
	clusters = pixels[cluster_inds.squeeze()]
	for i in range(0,K):
		pixel_ind = np.transpose(np.nonzero(assign_r[:, i]))
		quick = True
		if quick:  # using knowledge of L2 norm for faster solution
			cluster_ind = kmed_L2centroid(pixel_ind, pixels)
		else:
			cluster_ind = kmed_findcentroid(pixel_ind, pixels)
		cluster_inds[i] = cluster_ind

	clusters = pixels[cluster_inds.squeeze()]
	uniq, ind = np.unique(clusters, axis=0, return_index=True)
	while ind.shape[0] < clusters.shape[0]:
		for i in range(0, clusters.shape[0]):
			if i not in ind:
				clusters[i] = pixels[np.random.randint(0, pixels.shape[0])]
			uniq, ind = np.unique(clusters, axis=0, return_index=True)
	return clusters


def kmed_updateclusters_matrix(assign_r, pixels, K, D):
	cluster_inds = np.zeros((K,1), np.int)
	clusters = pixels[cluster_inds.squeeze()]
	for i in range(0,K):
		pixel_ind = np.transpose(np.nonzero(assign_r[:, i]))
		cluster_ind = kmed_findcentroid_matrix(pixel_ind, pixels, D)
		cluster_inds[i] = cluster_ind

	clusters = pixels[cluster_inds.squeeze()]
	uniq, ind = np.unique(clusters, axis=0, return_index=True)
	while ind.shape[0] < clusters.shape[0]:
		for i in range(0, clusters.shape[0]):
			if i not in ind:
				clusters[i] = pixels[np.random.randint(0, pixels.shape[0])]
		uniq, ind = np.unique(clusters, axis=0, return_index=True)
	return clusters


def mykmedoids(pixels, K, option=0):
	'''
	K-Mediod Options:
	0: Quick L2 norm implementation
	1: Slow L2 (TODO)
	2: Matrix Math L2 (TODO)
	'''
	centroids_ind = np.random.randint(0, pixels.shape[0], (K, 1))  # initialize as random index location

	centroids = pixels[centroids_ind.squeeze()]
	assignments_r = np.zeros((pixels.shape[0], K))

	# if matrix option, calculate D matrix
	if option == 2:
		D = distance_matrix(pixels)
	elif option == 3:
		D = distance_matrix_manhattan(pixels)
	elif option == 4:
		D = distance_matrix_chebychev(pixels)
	steps = 100
	cnt = 1
	epsilon = 1e-1
	converged = False
	print("Clustering iterations for K-Medoids are beginning...")
	while not converged:
		# CLUSTER ASSIGNMENT
		if option==2 or option == 3 or option == 4:
			assignments_r = kmed_assignclusters_matrix(pixels, centroids_ind, K, D)
		else:
			assignments_r = kmed_assignclusters(pixels, centroids, K, option)
		centroids_last = centroids
		# CLUSTER ADJUSTMENT
		if option==2 or option == 3 or option == 4:
			centroids = kmed_updateclusters_matrix(assignments_r, pixels, K, D)
		else:
			centroids = kmed_updateclusters(assignments_r, pixels, K)

		if (cnt % 10 == 0 or cnt == 1):
			print("K-Mediods Iteration #{}".format(cnt))
			print("Centroid clusters: \n {}".format(centroids))
		cnt = cnt + 1
		array, locations = np.where(assignments_r == 1)
		classes = locations.reshape((locations.shape[0], 1))
		if np.linalg.norm(centroids-centroids_last, axis=0).sum() < epsilon:
			converged = True
			print("K-Mediods Converged after {} iterations".format(cnt))
		elif cnt >= steps:
			converged = True
			print("K-Mediods Stopped after reaching max {} iterations".format(cnt))
		else:
			pass
	return classes, centroids


def main():

	if(len(sys.argv) < 2):
		print("Please supply an image file")
		return

	image_file_name = sys.argv[1]
	K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
	print(image_file_name, K)
	im = np.asarray(imageio.imread(image_file_name))

	plt.imshow(im)

	fig, axs = plt.subplots(1, 2)

	im_vector = im.copy()
	im_vector.resize((im.shape[0]*im.shape[1], im.shape[2])) # added

	t1 = time.time()
	print("Starting K-medoids Clustering")
	classes, centers = mykmedoids(im_vector, K, option=0)  # Options: 0: L2 quick, 2: Matrix L2, 3: Matrix L1, 4: Matrix Linf
	print(classes, centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[0].imshow(new_im)
	axs[0].set_title('K-medoids')
	t2 = time.time()
	kmed_time = (t2-t1)

	t1 = time.time()
	print("Starting K-means Clustering")
	classes, centers = mykmeans(im_vector, K)
	print(classes, centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[1].imshow(new_im)
	axs[1].set_title('K-means')
	t2 = time.time()
	kmean_time = (t2-t1)

	plt.show()

	print("Total times\n K-mediods: {} s \n K-Means: {} s".format(kmed_time, kmean_time))


if __name__ == '__main__':
	main()