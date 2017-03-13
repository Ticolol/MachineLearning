import numpy as np
import pandas as pd
import math
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor


def RunRegression(func, params, datas, labels):
	best_mae = float('inf')
	best_param = []
	ext_skf = KFold(5)
	for ext_tr, ext_te in ext_skf.split(datas,labels):
		d_tr, d_te = datas[ext_tr], datas[ext_te]
		l_tr, l_te = labels[ext_tr], labels[ext_te]
			
		ss = ShuffleSplit(n_splits=1, test_size=0.25)
		for index_tr, index_te in ss.split(d_tr):			
			tr, te = d_tr[index_tr], d_tr[index_te]
			ltr, lte = l_tr[index_tr], l_tr[index_te]	
			#print(tr.shape)
			#print(ltr.shape)
			#print(params)		
			reg = GridSearchCV(func, params)		
			reg.fit(tr, ltr)		
			pred = reg.predict(te)
			print("oOOOOOOOORAAAAAA")				
			mae = mean_absolute_error(lte, pred)
			if(mae < best_mae):
				best_mae = mae
				best_param = reg.best_params_

	print(best_mae)		 		
	print(best_param)		 		

#Preprocess data	
def Preprocess(datas):
	#Encode labels
	le = LabelEncoder()
	for col in range(datas.shape[1]):
		column = datas[:,col]
		if (type(column[0]) is not int) and (type(column[0]) is not float):
			column = le.fit_transform(column)
			datas[:,col] = column.astype(float)	
	#Remove low variance features
	threshold = 0.8
	sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))	
	datas = sel.fit_transform(datas)
	#Applying PCA
	pca = PCA( n_components =0.95)
	datas = pca.fit_transform(datas)
	#Normalization
	datas = normalize(datas)

	return datas.astype(float)


if __name__=='__main__':

	# Get data
	raw_data = pd.read_csv('train.csv', ',')
	#print(raw_data.shape)
	raw_test = pd.read_csv('test.csv', ',')
	#print(raw_test.shape)
	labels = raw_data.values[:,0].astype(int)
	datas = raw_data.values[:,1:]
	test = raw_test.values
	print(datas.shape)
	print(labels.shape)
	print(test.shape)


	svm_params = {'kernel': ['rbf'], 'C': [2**-5, 2**0, 2**5, 2**10], 'gamma': [2**-15, 2**-10, 2**-5, 2**0, 2**5]}
	gbc_params = {'learning_rate': [0.1, 0.05],'n_estimators': [1, 5, 11, 15, 21, 25], 'max_depth': [5]}
	rf_params = {'n_estimators': [100, 200, 300, 400],'max_features': [6,8,10]}
	rn_params = {'hidden_layer_sizes': [10, 20, 30, 40]}
	gr_params = {'alpha': [10**-12,10**-10,10**-8,10**-6]}

	print("=====> Preprocessing")
	datas = Preprocess(datas)
	test = Preprocess(test)
	print(datas.shape)

	print("=====> Support Vector Regression")	
	RunRegression(SVR(), svm_params, datas, labels)

	print("=====> Gradient Boosting Regressor")	
	RunRegression(GradientBoostingRegressor(), gbc_params, datas, labels)	

	print("=====> Random Forest")		
	RunRegression(RandomForestRegressor(), rf_params, datas, labels)	

	print("=====> Neural Networks")
	RunRegression(MLPRegressor(), rn_params, datas, labels)	

	print("=====> Gaussian Process Regressor")		
	#Hyper parameters must be revised
	RunRegression(GaussianProcessRegressor(), gr_params, datas, labels)	

	#Best time: MLPRegressor
	final = MLPRegressor(hidden_layer_sizes=10)		
	final.fit(datas,labels)
	pref = final.predict(test)
	for x in pref:
		print(x)


