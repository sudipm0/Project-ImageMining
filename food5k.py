# -*- coding: utf-8 -*-
'''Data loading module.

'''

from __future__ import print_function

import os
import numpy as np
from tensorflow.python.keras import backend as K
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input


def load_data():
	cwd = os.getcwd()
	path = cwd+'\\images\\'
	saveddatapath ='SavedData\\Food5k'
	trainingdir = 'Food-5K\\training'
	testdir = 'Food-5K\\validation'

	testfolder = os.path.join(path, testdir)
	saveddatafolder = os.path.join(path, saveddatapath)
	trainfolder = os.path.join(path,trainingdir)
	num_test_samples = sum([len(files) for r, d, files in os.walk(testfolder)])  #len([name for name in os.listdir(testfolder) if os.path.isfile(os.path.join(testfolder,name))])
	num_train_samples = sum([len(files) for r, d, files in os.walk(trainfolder)])

	print('Number of training samples: ', num_train_samples)
	print('Number of test samples: ', num_test_samples)

	x_train = np.empty((num_train_samples, 224, 224,3), dtype='uint8') #This should be increased.
	x_test = np.empty((num_test_samples, 224, 224,3), dtype='uint8')	
	y_test = np.empty((num_test_samples), dtype='int8')

	#Load up training data samples
	for i in range(num_train_samples):
		try:
			img_path = os.path.join(path,trainingdir, '1_' + str(i)+'.jpg')
			#print('Image path: ' + img_path)
			img = image.load_img(img_path, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			x_train[(i-1):i, :, :, :] = x
		except FileNotFoundError:
			print("The file does not exist",img_path)


	#Load up test data sample from testdir
	i = 0
	for path, dirs, files in os.walk(testfolder):
		for file in files:
			if file.endswith(".jpg"):
				label = file[0:file.find("_")]
				if label == '1':
					label = 1
				else:
					label = -1
				img_path = os.path.join(path, file)
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				x_test[(i-1):i, :, :, :] = x
				y_test[(i-1) :i] = label
				i += 1
	print('Saving Data arrays')
	if not os.path.exists(saveddatafolder):
		os.makedirs(saveddatafolder)
	np.save(os.path.join(saveddatafolder,'x_train'),x_train,allow_pickle=True)
	np.save(os.path.join(saveddatafolder,'x_test'),x_test,allow_pickle=True)
	np.save(os.path.join(saveddatafolder,'y_test'),y_test,allow_pickle=True)

	return (x_train,num_train_samples,x_test,y_test,num_test_samples)

if __name__ == '__main__':
	(x_train,x_test,y_test) = load_data()
	print('Test Label1 :', y_test)