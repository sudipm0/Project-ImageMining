from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, make_scorer

def GridSearch(X,Y,estimator_str):

	ss = StandardScaler()
	ss.fit(X)
	X = ss.transform(X)

	# Take PCA to reduce feature space dimensionality
	pca = PCA(n_components=512, whiten=True)
	pca = pca.fit(X)	
	X = pca.transform(X)
	
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
	f1sc = make_scorer(f1_score(y_train, y_test, average='micro')) 

	if estimator_str == 'svc':		
		parameters = {'kernel':('linear', 'rbf'),'C':[10],'gamma': [1e-3, 1e-4]}		
		estimator = svm.SVC()
		clf = GridSearchCV(estimator, parameters)
		clf.fit(X, Y)
		
	elif estimator_str == 'isolationforest':
		estimator = IsolationForest()
		estimator.contamination = sum(Y==-1)/len(Y)
		parameters = {"n_estimators": (40, 55, 75, 95, 115)}
		clf = GridSearchCV(estimator, param_grid=parameters,scoring=f1sc)
		clf.fit(X_train, y_train)
	
	print("Best parameters set found on train set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on train set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
		print()

	#Let's experiment on best_params
	return(clf.best_params_)
	

#if __name__ == '__main__':
	#GridSearch()