# -*- coding: utf-8 -*-
'''Data loading module.

'''

from __future__ import print_function

import os
import numpy as np
import csv
from tensorflow.python.keras import backend as K
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
	cwd = os.getcwd()
	plt.ioff()
	path = cwd+'\\images\\'
	saveddatapath ='SavedData\\Food5k'
	trainingdir = 'Food-5K\\training\\food'
	testdir = 'Food-5K\\validation'
	testoutputfile = 'test_data_food5k.csv'

	testfolder = os.path.join(path, testdir)
	saveddatafolder = os.path.join(path, saveddatapath)
	testoutputfile = os.path.join(saveddatafolder, testoutputfile)
	trainfolder = os.path.join(path,trainingdir)
	num_test_samples = sum([len(files) for r, d, files in os.walk(testfolder)])  #len([name for name in os.listdir(testfolder) if os.path.isfile(os.path.join(testfolder,name))])
	num_train_samples = sum([len(files) for r, d, files in os.walk(trainfolder)])

	print('Number of training samples: ', num_train_samples)
	print('Number of test samples: ', num_test_samples)

	x_train = np.empty((num_train_samples, 224, 224,3), dtype='uint8')
	x_test = np.empty((num_test_samples, 224, 224,3), dtype='uint8')	
	y_train = np.empty((num_train_samples), dtype='int8')
	y_test = np.empty((num_test_samples), dtype='int8')

	if not os.path.exists(saveddatafolder):
		os.makedirs(saveddatafolder)

	# #Load up training data samples
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
	with open(testoutputfile, mode='w') as test_file:
		writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
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
					writer.writerow([img_path, '1'])	#let's write everything positive
					i += 1	
		
	print('Saving Data arrays')
	
	np.save(os.path.join(saveddatafolder,'x_train'),x_train,allow_pickle=True)
	np.save(os.path.join(saveddatafolder,'x_test'),x_test,allow_pickle=True)
	np.save(os.path.join(saveddatafolder,'y_test'),y_test,allow_pickle=True)
	
	df = pd.read_csv(testoutputfile, names=['Images', 'Class'])
	df = df.groupby(['Class']).size().reset_index(name='ClassCount')
	output_img_file_name = 'testdata.png'
	df.plot(kind='bar',x='Class',y='ClassCount',color='green',title ='Test Data for reference')	
	plt.savefig(os.path.join(saveddatafolder,output_img_file_name))

	return (x_train,y_train,num_train_samples,x_test,y_test,num_test_samples,testoutputfile)

if __name__ == '__main__':
	(x_train,x_test,y_test) = load_data()
	print('Test Label1 :', y_test)