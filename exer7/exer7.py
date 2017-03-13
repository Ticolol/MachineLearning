import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt


'''

=======> Trabalho 7 - Deteccao de anomalias

	- Neste trabalho, visei encontrar os erros através de um sistema que separa o espaço de dados em blocos e
	calcula os desvios padroes de cada um destes intervalos. Obtidos este conjunto de valores, eh calculado o
	desvio padrao deste conjunto de desvios padroes. 

	- Obtido o desvio padrao dos desvios padroes, inicia-se a etapa de CLUSTERIZAÇÃO DOS DESVIOS PADROES. Se
	dado valor deste conjunto se encontra dentro de um intervalo de um cluster, o valor se une ao cluster e 
	este tem sua media atualizada. Caso contrario, cria-se um novo cluster para este valor. Um intervalo de 
	um cluster é calculado a partir da media dos valores do cluster e do desvio padrao dos desvios padroes, 
	calculado anteriormente

	- Após a clusterização dos blocos, ve-se a quantidade de elementos em cada cluster. Um cluster com poucos 
	elementos constitui um comportamento não frequente, possivelmente anormal e, consequentemente, anômalo

=====> COMO USAR
	
	- Execute no console:

		python exer7.py <serieX.csv>

'''

if __name__=='__main__':
	#READ
	# Get data
	raw_data = pd.read_csv(sys.argv[1], ',')
	datas = raw_data.values[:,1:]
	#print(datas.shape)
	#print(datas)

	# Set number of blocks
	N=21
	size=datas.shape[0]/N#192
	anomalySize = N/7

	# Calculate means and standard deviations
	means=[]
	stds=[]
	for x in range(N):		
		means.append(np.mean(datas[int(size*x):int(size*(x+1))]))
		stds.append(np.std(datas[int(size*x):int(size*(x+1))]))



	#print(np.mean(datas))
	#print(np.std(datas))

	#print(means)
	print(stds)
	print(np.std(stds))
	#print(np.mean(stds))

	#Get standard deviation from standard deviations
	stdMean=np.std(stds)
	clustersMean=[]
	clustersSize=[]
	clusterIntervals=[]
	clustersMean.append(stds[0])
	clustersSize.append(1)	
	clusterIntervals.append([0])
	for x in range(1,N):
		val = stds[x]
		
		c=-1
		#Check for each existing cluster
		for y in range(len(clustersMean)):
			clu = clustersMean[y]
			# Check if value falls under current cluster
			if(clu-stdMean*1.5 <= val <= clu+stdMean*1.5):
				#Belongs to this cluster
				c = y
				break
		#Check if any cluster was matched
		if(c!=-1):
			#Found a cluster
			clustersMean[c]=((clustersMean[c]*clustersSize[c])+val)/(clustersSize[c]+1)
			clustersSize[c]+=1
			clusterIntervals[c].append(x)
		else:
			#No matching cluster found. Create new cluster
			clustersMean.append(val)
			clustersSize.append(1)
			clusterIntervals.append([x])
		
	print(clustersMean)
	print(clustersSize)
	print(clusterIntervals)

	#If cluster is too small, it's probably an anomaly
	anomalyInterval=[]
	for x in range(len(clustersMean)):
		if(clustersSize[x]<=anomalySize):
			#It's an anomaly point
			anomalyInterval.extend(clusterIntervals[x])

	print(anomalyInterval)
	anomalyInterval=[[y*size] for y in anomalyInterval]

	#Plot graph
	plt.figure(1)
	plt.clf()
	
	plt.plot(datas)
	for a in anomalyInterval:
		plt.axvline(a,color='r')
	#plt.plot(anomalyInterval, reduced_data[:, 1], 'k.', markersize=2)

	#centroids = km2.cluster_centers_
	#plt.scatter(centroids[:, 0], centroids[:, 1],
    #        marker='x', s=169, linewidths=3,
    #        color='b', zorder=10)
	plt.title('Anomaly detection\n')
	#plt.xlim(x_min, x_minmax)
	#plt.ylim(y_min, y_max)
	#plt.xticks(())
	#plt.yticks(())
	plt.show()



		

