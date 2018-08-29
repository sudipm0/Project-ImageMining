from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from anomalydata import load_anomaly_data

def GridSearch():
	(x_train,num_train_samples,x_test,y_test,num_test_samples,testoutputfile) = load_anomaly_data()
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svc = svm.SVC()
	clf = GridSearchCV(svc, parameters)	
	print(x_test.shape)
	print(y_test.shape)
	
	clf.fit(x_test, y_test)
	print(clf.cv_results_.keys())
	return(clf)
	

if __name__ == '__main__':
	GridSearch()