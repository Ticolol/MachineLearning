import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA
from sklearn import metrics


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 1     # point in the mesh [x_min, x_max]x[y_min, y_max].


if __name__=='__main__':

	# Get data
	raw_data = pd.read_csv('cluster-data.csv', ',')
	raw_labels = pd.read_csv('cluster-data-class.csv', ' ')
	datas = raw_data.values
	labels = raw_labels.values

	print(datas.shape)


	# ============ Internal metric: Dunn 
	print("=> Starting Kmeans with internal metric using Dunn of distances")
	# Get best value for k
	best_dunn = 0
	best_k = 2
	for k in range(2,11):		
		inter, intra = 0, 0
		km = KMeans(n_clusters=k, n_init=5)
		km.fit(datas)
		# Calculate Dunn Index of center distances
		dist = DistanceMetric.get_metric('euclidean')
		# INTRA
		intraGroup = []
		for i in range(k):
			intraGroup.append(0)
		for i in range(len(datas)):
			X = [datas[i],km.cluster_centers_[km.labels_[i]]]
			intraGroup[km.labels_[i]] += float(dist.pairwise(X)[0,1])		
		intra = max(intraGroup)			
		#INTER
		interGroup = []
		for i in range(k): # For each cluster
			for j in range(i+1,k):
				X=[km.cluster_centers_[i], km.cluster_centers_[j]]
				interGroup.append(float(dist.pairwise(X)[0,1]))
		inter = min(interGroup)			
		#DUNN
		dunn=inter/intra
		if(dunn>best_dunn):
			best_dunn = dunn
			best_k = k

	print("- Best index found: " + str(best_dunn))
	print("- Value for k: " + str(best_k))
	

	# Plot
	reduced_data = PCA(n_components=2).fit_transform(datas)
	km = KMeans(n_clusters=best_k, n_init=5)
	km.fit(reduced_data)
	
	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
	y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Obtain labels for each point in mesh. Use last trained model.
	Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

	plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

	centroids = km.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)	
	plt.title('K-means - Internal evaluation by Dunn Index of distances (sum of distances) \n(data reduction via PCA)')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()


	# ============ External metric: Indice Rand
	print("=> Starting Kmeans with external metric using Rand Index")
	# Get best value for k
	best_score = 0
	best_k = 2

	for k in range(2,11):		
		km2 = KMeans(n_clusters=k, n_init=5)
		km2.fit(datas)
		score = metrics.adjusted_rand_score(labels[:,0], km2.labels_)
		if(score>best_score):
			best_score = score
			best_k = k

	print("- Best index found: " + str(best_score))
	print("- Value for k: " + str(best_k))

	# Plot
	reduced_data = PCA(n_components=2).fit_transform(datas)
	km2 = KMeans(n_clusters=best_k, n_init=5)
	km2.fit(reduced_data)

	
	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
	y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Obtain labels for each point in mesh. Use last trained model.
	Z = km2.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)

	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Paired,
	           aspect='auto', origin='lower')
	
	plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

	centroids = km2.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)
	plt.title('K-means - External evaluation by Rand Index (data reduction via PCA)\n')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()

	