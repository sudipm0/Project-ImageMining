from __future__ import print_function

import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time
import csv
import numpy as np

def Analyze(estimator_str,estimator_params,X_train,y_train, X_test,y_test,testoutputfile):
	tic = time.clock()
	print('Starting analysis...')
	
	ss = StandardScaler()
	ss.fit(X_train)
	X_train = ss.transform(X_train)
	X_test = ss.transform(X_test)

	# Take PCA to reduce feature space dimensionality
	pca = PCA(n_components=512, whiten=True)
	pca = pca.fit(X_train)
	
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	
	if estimator_str == "isolationforest":
		if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)
		if_clf.fit(X_train)
		predictions = if_clf.predict(X_test)
	else:	
		oc_svm_clf = svm.OneClassSVM(gamma=estimator_params['gamma'], kernel=estimator_params['kernel'], nu=0.05)
		oc_svm_clf.fit(X_train)
		predictions = oc_svm_clf.predict(X_test)

	#update results to test data output file	
	UpdateOutput(predictions,testoutputfile)
	#updating test results complete
	
	if estimator_str == "isolationforest":
		print('-------------------Isolation Forest Test Matrix---------------------------------------')
		Accuracy_Score = accuracy_score(y_test, predictions)	
		print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
		print('------------------------END----------------------------------------')
	else:
		print('-------------------ONE Class SVM Test Matrix---------------------------------------')
		Accuracy_Score = accuracy_score(y_test, predictions)	
		print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
		print('------------------------END----------------------------------------')
	toc = time.clock()
	print("Total time taken for analysis = ", toc-tic)

def UpdateOutput(predictions,testoutputfile):
	testresults = []
	with open(testoutputfile,'r') as testcsvfile:
		reader = csv.reader(testcsvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			testresults.append(row)

	index = 0
	modifiedtestresults = []
	for result in np.nditer(predictions):		
		if(len(testresults) > index):			
			row = testresults[index]
			row[1] = result
			modifiedtestresults.append(row)
		index += 1
	#print("total items ",index)

	with open(testoutputfile, mode='w') as testcsvfile:
		writer = csv.writer(testcsvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
		for row in modifiedtestresults:
			writer.writerow(row)

	