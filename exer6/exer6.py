import numpy as np
import pandas as pd
import math
import string
from sklearn.datasets import load_files
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC



categories = ['Apps','Social','Enterprise','Gadgets','Startups']


def pred_accuracy(pred, lte):
	hits = [None] * len(pred)
	for i in range(0,len(pred)):
		if pred[i] == lte[i]:
			hits[i] = 1
		else:
			hits[i] = 0
	return sum(hits)/len(pred)

def SVMkernelRBF (datas, labels):
	print(" => Starting SVM using RBF kernel with 5x3-Fold and C and Gamma as HiperParameters")
	aver_accur = 0
	best_accur = 0
	best_C = 1
	best_gamma = 1
	ext_skf = KFold(5)
	for ext_tr, ext_te in ext_skf.split(datas,labels):
		d_tr, d_te = datas[ext_tr], datas[ext_te]
		l_tr, l_te = labels[ext_tr], labels[ext_te]
		
		int_ac = 0
		int_C = 1
		int_gamma = 1		

		int_skf = KFold(3)
		for int_tr, int_te in int_skf.split(d_tr,l_tr):			
			tr, te = d_tr[int_tr], d_tr[int_te]
			ltr, lte = l_tr[int_tr], l_tr[int_te]		

			for C in c_values:
				for gamma in gamma_values:
					#Training
					clf = SVC(C=C, kernel='rbf', gamma=gamma)					
					clf.fit(tr, ltr.astype(int))	
					#Testing
					pred = clf.predict(te)
					ac = pred_accuracy(pred, lte)
					#Check if result's the best
					if ac > int_ac:
						int_ac, int_C, int_gamma = ac, C, gamma

		#Training
		clf = SVC(C=int_C, kernel='rbf', gamma=int_gamma)
		clf.fit(d_tr, l_tr.astype(int))			
		#Testing
		pred = clf.predict(d_te)
		ac = pred_accuracy(pred, l_te)			
		print("Accuracy="+str(ac)+"; C="+str(int_C)+"; gamma="+str(int_gamma))
		aver_accur += ac
		if ac>best_accur:
			best_accur, best_C, best_gamma = ac, int_C, int_gamma

	aver_accur /= 5
	print("=========================")
	print("Average accuracy="+str(aver_accur))
	print("=========================")
	print("")

def gradientBoostingClassifier(datas, labels):
	tree_breadth = 5
	print(" => Starting Gradient Boosting Classifier with 5x3-Fold and nTrees and Learning rate as HiperParameters")
	aver_accur = 0
	best_accur = 0
	ext_skf = KFold(5)
	for ext_tr, ext_te in ext_skf.split(datas,labels):
			d_tr, d_te = datas[ext_tr], datas[ext_te]
			l_tr, l_te = labels[ext_tr], labels[ext_te]

			int_ac = 0			
			int_lr = 0
			int_trees = 0

			int_skf = KFold(3)
			for int_tr, int_te in int_skf.split(d_tr,l_tr):			
				tr, te = d_tr[int_tr], d_tr[int_te]
				ltr, lte = l_tr[int_tr], l_tr[int_te]		
				#print("Fold")
				for lr in gbm_lr:
					for t in gbm_trees:
						gbm = GradientBoostingClassifier(loss='deviance',learning_rate=lr, n_estimators=t, max_depth=tree_breadth)
						gbm.fit(tr, ltr.astype(int))
						#Testing
						pred = gbm.predict(te)
						ac = pred_accuracy(pred, lte)
						#Check if result's the best
						if ac > int_ac:
							int_ac, int_trees, int_lr = ac, t, lr

			#Training
			gbm = GradientBoostingClassifier(loss='deviance',learning_rate=lr, n_estimators=t, max_depth=tree_breadth)
			gbm.fit(d_tr, l_tr.astype(int))			
			#Testing
			pred = gbm.predict(d_te)
			ac = pred_accuracy(pred, l_te)
			print("Accuracy="+str(ac)+"; Number of trees="+str(int_trees)+"; Learning Rate="+str(int_lr))
			aver_accur += ac
			if ac>best_accur:
				best_accur  = ac

	aver_accur /= 5
	print("=========================")
	print("Average accuracy="+str(aver_accur))
	print("=========================")
	print("")


if __name__=='__main__':
	#READ
	# Ler labels
	print(" > Reading labels")
	raw_data = pd.read_csv('category.tab',' ')
	labels = raw_data.values
	# Ler textos
	print(" > Reading files")
	v_treinos = load_files(container_path='filesk/',encoding='utf-8')
	print(len(v_treinos.data))	
	#print("\n".join(v_treinos.data[0].split("\n")[:3]))

	#PREPROCESSING PT1
	print(" > Preprocessing pt1")
	for l in range(len(v_treinos.data)):
		# Lowercase
		line = v_treinos.data[l].lower()
		# Remove punctuation
		line = line.translate(str.maketrans({key: None for key in string.punctuation}))
		# Stemming
		singles = []
		stemmer = PorterStemmer() 
		for plural in line.split():
			singles.append(stemmer.stem(plural))
		line = ' '.join(singles)

		v_treinos.data[l] = line
	#print((v_treinos.data[0]))

	#BAG OF WORDS (through CountVectorizer)
	print(" > Creating bag of words")
	vec = CountVectorizer(stop_words='english')
	bag = vec.fit_transform(v_treinos.data)	
	print(len(vec.get_feature_names()))

	#GET O/1 BAG OF WORDS - Tentar otimizar esta etapa	
	print(" > Creating binary bag of words")
	bag_01= bag.copy()	
	A = coo_matrix(bag_01)
	for i,j,v in zip(A.row, A.col, A.data):
		if(v!=0):
			bag_01[i,j]=1

	#PREPROCESSING PT2
	print(" > Preprocessing pt2")

	#line = PorterStemmer().stem_word(line)

	#====================
	# PARTE 2
	#====================
	print(" === Part 2")
	
	bag = bag.todense()
	bag_01 = bag_01.todense()
	
	# Separação em Kfolds
	print(" > Making 5 Fold")
	kf = KFold(5)
	for ext_tr, ext_te in kf.split(labels):
		d_tr, d_te = bag[ext_tr], bag[ext_te]
		db_tr, db_te = bag_01[ext_tr], bag_01[ext_te]
		l_tr, l_te = labels[ext_tr], labels[ext_te]

		#Naive Bayes
		gnb = GaussianNB()
		gnb.fit(db_tr, l_tr)		
		pred = gnb.predict(db_te)
		ac = pred_accuracy(pred, l_te)
		print("Naive Bayes Accuracy="+str(ac))

		#Logistic Regression: Binary
		model = LogisticRegression(C=10000)
		model.fit(db_tr, l_tr)
		pred = model.predict(db_te)
		ac = pred_accuracy(pred, l_te)
		print("LR Binary Accuracy="+str(ac))

		#Logistic Regression: Term Frequency
		model2 = LogisticRegression(C=10000)
		model2.fit(d_tr, l_tr)
		pred = model2.predict(d_te)
		ac = pred_accuracy(pred, l_te)
		print("LR Term Frequency Accuracy="+str(ac))


	#====================
	# PARTE 3
	#====================
	print(" === Part 3")

	#Applying PCA
	print(" > Applying PCA on data, maintaining 99% of variance")	
	pca = PCA()
	pca.fit(bag)
	var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))	
	x=0
	while (var[x] < 0.99):
		x+=1
	x+=1
	pca = PCA(n_components=x)
	datas_t = pca.fit_transform(bag)

	#RUN SVM with RBF Kernel
	print(" > Running SVM with RBF kernel")	
	SVMkernelRBF (datas_t, labels)

	#RUN GBM
	print(" > Running GBM")	
	GradientBoostingClassifier (datas_t, labels)
