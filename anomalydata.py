# -*- coding: utf-8 -*-
'''Data loading module.

'''
from __future__ import print_function

import os
import numpy as np
from tensorflow.python.keras import backend as K
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input


def load_anomaly_data():
	cwd = os.getcwd()
	path = cwd+'\\images\\'
	saveddatapath ='SavedData\\AnomalyData'
	trainingdir = 'VOC2007\\JPEGImages'
	testdir = 'Abnormal_Object_Dataset'

	testfolder = os.path.join(path, testdir)
	trainfolder = os.path.join(path, trainingdir)
	saveddatafolder = os.path.join(path, saveddatapath)
	num_test_samples = sum([len(files) for r, d, files in os.walk(testfolder)])
	num_train_samples = sum([len(files) for r, d, files in os.walk(trainfolder)])
	
	print('Number of training samples: ', num_train_samples)
	print('Number of test samples: ', num_test_samples)
	
	x_train = np.empty((num_train_samples, 224, 224,3), dtype='uint8') #This should be increased.
	x_test = np.empty((num_test_samples, 224, 224,3), dtype='uint8')	
	y_test = np.empty((num_test_samples), dtype='int8')

	j = 0
	for path, subdirs, files in os.walk(trainfolder):
		for file in files:
			if file.endswith(".jpg"):				
				img_path = os.path.join(path, file)
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				x_train[(j-1):j, :, :, :] = x
				j += 1

	i = 0
	for path, subdirs, files in os.walk(testfolder):
		for file in files:
			if file.endswith(".jpg"):								
				img_path = os.path.join(path, file)
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				x_test[(i-1):i, :, :, :] = x
				y_test[(i-1):i] = -1				
				i += 1
	
	return (x_train,num_train_samples,x_test,y_test,num_test_samples)

if __name__ == '__main__':
	(x_train,x_test,y_test) = load_anomaly_data()
	#print('Test Label1 :', y_test)