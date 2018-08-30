""" 
Project: Effifient Image Mining & Classification Techniques
Project Goal:
  The project aims to study on below topics/problems:
    1. If a new image is added to a folder, is it causing anomaly in the existing dataset? e.g. training dataset is already labeled
    2. IF a model is trained with a set of images can it detect contamination/anomaly in another folder?
    3. Cluster images by their feature set.

Approach to the solution(s):
  1. We use CNN pre-trained models to extract features from images. By default, we choose ResNet50 in this case. 
  We exclude the classification/prediction layer.

  2. Dimension of feature space matrix of shape e.g. (1L,1L,1L) is then reduced to n_components using PCA.

  Now For Novelty & Outlier detection, we choose OneClassSVM & IsolationForest.

  OneClassSVM is used as Supervised training method/tool where environment is trained with only positive data points
  and once a data point which shouldn't belong to the current set is seen, it detects an anomaly.


This is the driver program for the project.
rus as:
  python driver.py
  
  [arguments]   [default]   [options]
   --backbone     resnet      resnet/vgg
   --dataset      food5k      food5k/PascalVOC
   --task         anomaly     anomaly/cluster
   --subtask      novelty     novelty/outlier

"""

import warnings
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import argparse

from analyze import Analyze
from food5k import load_data
from anomalydata import load_anomaly_data
from clustering import Cluster
from gridsearch import GridSearch

def arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", default="resnet",help="Specify the backbone: vgg/resnet")
  parser.add_argument("--dataset", default="food5k",help="Specify the dataset: food5k/PascalVOC")
  parser.add_argument("--task", default="anomaly",help="Specify the task: anomaly/cluster")
  parser.add_argument("--subtask", default="novelty",help="Specify the subtask: novelty/outlier")
  return(parser)

if __name__ == '__main__':

  parser = arguments()
  args = parser.parse_args()
  
  if args.dataset == "PascalVOC" :
    (x_train,y_train,num_train_samples,x_test,y_test,num_test_samples,testoutputfile) = load_anomaly_data()
  else:
    (x_train,y_train,num_train_samples,x_test,y_test,num_test_samples,testoutputfile) = load_data()

  if args.backbone == "vgg":
    model = VGG16(include_top=False, weights='imagenet')
    #model_vgg = Model(input = base_vgg_model.input, output = base_vgg_model.get_layer('block4_pool').output)    
  else:
    model = ResNet50(weights='imagenet', include_top=False)

  if args.task == "anomaly":
    features_train_array = model.predict(x_train)
    features_train_array = features_train_array.reshape(num_train_samples, -1) #reshape to 2d from 4d array
      
    features_test_array = model.predict(x_test)
    features_test_array = features_test_array.reshape(num_test_samples, -1)
    estimator_str = "svc"
    if args.subtask == "outlier":
      estimator_str = "isolationforest"

    estimator_params = GridSearch(features_train_array,y_train,estimator_str)

    Analyze(estimator_str,estimator_params,features_train_array,y_train,features_test_array,y_test,testoutputfile)
   

  if args.task == "cluster":
    features_train_array = model.predict(x_train)
    features_train_array = features_train_array.reshape(num_train_samples, -1) #reshape to 2d from 4d array
      
    #features_test_array = model.predict(x_test)
    #features_test_array = features_test_array.reshape(num_test_samples, -1)    
    Cluster(features_train_array,num_train_samples)






