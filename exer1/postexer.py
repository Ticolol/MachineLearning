import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn import metrics

data = pd.read_csv('data1.csv')
array = data.values

X = array[:,0:165]
Y = array[:,166]

#print(X)
#print(Y)

pca = PCA()
pca.fit(X)
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var)

pca = PCA(n_components=13)
X_transf = pca.fit_transform(X)


model = LogisticRegression()
model = model.fit(X[:200,:],Y[:200])
model_transf = LogisticRegression()
model_transf = model_transf.fit(X_transf[:200,:],Y[:200])

predicted = model.predict(X[200:])
predicted_transf = model_transf.predict(X_transf[200:])


print("Acuracia da regressao logistica no conjunto de dados original: "+ \
	str(metrics.accuracy_score(Y[200:], predicted)))
print("Acuracia da regressao logistica no conjunto de dados transformado: "+ \
	str(metrics.accuracy_score(Y[200:], predicted_transf)))


model_LDA = LDA()
model_LDA = model_LDA.fit(X[:200],Y[:200])
model_LDA_transf = LDA()
model_LDA_transf = model_LDA_transf.fit(X_transf[:200],Y[:200])
predicted_LDA = model_LDA.predict(X[200:])
predicted_LDA_transf = model_LDA_transf.predict(X_transf[200:])

print("Acuracia do LDA no conjunto de dados original: "+ \
	str(metrics.accuracy_score(Y[200:], predicted_LDA)))
print("Acuracia do LDA no conjunto de dados transformado: "+ \
	str(metrics.accuracy_score(Y[200:], predicted_LDA_transf)))









