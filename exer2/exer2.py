import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import sklearn


c_values = [2**-5, 2**-2, 2**0, 2**2, 2**5]
gamma_values = [2**-15, 2**-10, 2**-5, 2**0, 2**5]


def pred_accuracy(pred, lte):
	hits = [None] * len(pred)
	for i in range(0,len(pred)):
		if pred[i] == lte[i]:
			hits[i] = 1
		else:
			hits[i] = 0
	return sum(hits)/len(pred)


if __name__=='__main__':

	raw_data = pd.read_csv('data1.csv')
	raw_array = raw_data.values

	datas = raw_array[:,0:165]
	labels = raw_array[:,166]
	
	print(datas.shape)
	print(labels.shape)
	
	aver_accur = 0
	int_C = 1
	int_gamma = 1
	best_accur = 0
	best_C = 1
	best_gamma = 1

	#External stratified 5-fold
	print("Starting stratified 5-Fold with internal 3-Fold and Grid Search for HiperParameters")
	ext_skf = StratifiedKFold(5)
	for ext_tr, ext_te in ext_skf.split(datas,labels):
		#Preparations for internal k-fold
		d_tr, d_te = datas[ext_tr], datas[ext_te]
		l_tr, l_te = labels[ext_tr], labels[ext_te]
		int_ac = 0
		int_C = 1
		int_gamma = 1

		#Obtain hiperparameters: 3-fold training
		ext_skf = StratifiedKFold(3)
		for int_tr, int_te in ext_skf.split(d_tr,l_tr):
			#Preparations for training			
			tr, te = d_tr[int_tr], d_tr[int_te]
			ltr, lte = l_tr[int_tr], l_tr[int_te]		
			
			### Grid Search through hiperparameters C and gamma
			for C in c_values:
				for gamma in gamma_values:
					#Training
					clf = SVC(C=C, kernel='rbf', gamma=gamma)
					clf.fit(tr, ltr)			
					#Testing
					pred = clf.predict(te)
					ac = pred_accuracy(pred, lte)
					#Check if result's the best
					if ac > int_ac:
						int_ac, int_C, int_gamma = ac, C, gamma
		#Training
		clf = SVC(C=int_C, kernel='rbf', gamma=int_gamma)
		clf.fit(d_tr, l_tr)			
		#Testing
		pred = clf.predict(d_te)
		ac = pred_accuracy(pred, l_te)	
		print("C="+str(int_C)+"; gamma="+str(int_gamma))
		print("Accuracy="+str(ac))
		aver_accur += ac
		if ac>best_accur:
			best_accur, best_C, best_gamma = ac, int_C, int_gamma
		
	#Get average accuracy
	aver_accur /= 5
	print("=========================")
	print("Average accuracy="+str(aver_accur))
	print("=========================")
	print("")
		
	#Final 3-fold 	
	aver_accur = 0
	print(">>Starting final 3-Fold")
	print(">>>>C="+str(best_C)+"; gamma="+str(best_gamma))
	fin_skf = StratifiedKFold(3)
	for tr, te in fin_skf.split(datas,labels):
		#Training
		clf = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
		clf.fit(datas[tr], labels[tr])			
		#Testing
		pred = clf.predict(datas[te])
		ac = pred_accuracy(pred, labels[te])	
		print(">>Accuracy="+str(ac))
		aver_accur += ac
	aver_accur /= 3
	print("=========================")
	print("Final average accuracy="+str(aver_accur))
	print("=========================")
		

#Melhor resultado: C=4; gamma=0.03125

		
		
		
