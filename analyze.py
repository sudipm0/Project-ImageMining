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

def Analyze(X_train, X_test,y_test,testoutputfile):
	tic = time.clock()
	print('Starting analysis...')
	
	ss = StandardScaler()
	ss.fit(X_train)
	X_train = ss.transform(X_train)
	X_test = ss.transform(X_test)

	# Take PCA to reduce feature space dimensionality
	pca = PCA(n_components=512, whiten=True)
	pca = pca.fit(X_train)
	#print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	# Train classifier and obtain predictions for OC-SVM
	oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.08)  # Obtained using grid search
	if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

	oc_svm_clf.fit(X_train)
	if_clf.fit(X_train)

	oc_svm_preds = oc_svm_clf.predict(X_test)
	if_preds = if_clf.predict(X_test)
	
	testresults = []
	with open(testoutputfile,'r') as testcsvfile:
		reader = csv.reader(testcsvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			testresults.append(row)

	index = 0
	modifiedtestresults = []
	for result in np.nditer(oc_svm_preds):		
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

	print('-------------------ONE Class SVM Test Matrix---------------------------------------')
	Accuracy_Score = accuracy_score(y_test, oc_svm_preds)
	Precision_Score = precision_score(y_test, oc_svm_preds,  average="macro")
	Recall_Score = recall_score(y_test, oc_svm_preds,  average="macro")
	F1_Score = f1_score(y_test, oc_svm_preds,  average="macro")
	print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
	print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
	print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
	print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
	
	print('------------------------END----------------------------------------')

	print('-------------------Isolation Forest Test Matrix---------------------------------------')
	Accuracy_Score = accuracy_score(y_test, if_preds)
	Precision_Score = precision_score(y_test, if_preds,  average="macro")
	Recall_Score = recall_score(y_test, if_preds,  average="macro")
	F1_Score = f1_score(y_test, if_preds,  average="macro")
	print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
	print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
	print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
	print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
	
	print('------------------------END----------------------------------------')
	toc = time.clock()
	print("Total time taken for analysis = ", toc-tic)
	# Further compute accuracy, precision and recall for the two predictions sets obtained