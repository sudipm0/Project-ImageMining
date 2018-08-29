from __future__ import print_function

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

def arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", default="resnet",help="Specify the backbone: vgg/resnet")
  parser.add_argument("--dataset", default="food5k",help="Specify the backbone: food5k/PascalVOC")
  parser.add_argument("--task", default="anomaly",help="Specify the backbone: anomaly/cluster")
  return(parser)

if __name__ == '__main__':

  parser = arguments()
  args = parser.parse_args()
  
  if args.dataset == "PascalVOC" :
    (x_train,num_train_samples,x_test,y_test,num_test_samples) = load_anomaly_data()
  else:
    (x_train,num_train_samples,x_test,y_test,num_test_samples) = load_data()

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
    # print('test array shape: ',features_test_array.shape)
    # print('train array shape: ',features_train_array.shape)
    Analyze(features_train_array,features_test_array,y_test)
   

  if args.task == "cluster":
    features_train_array = model.predict(x_train)
    features_train_array = features_train_array.reshape(num_train_samples, -1) #reshape to 2d from 4d array
      
    #features_test_array = model.predict(x_test)
    #features_test_array = features_test_array.reshape(num_test_samples, -1)    
    Cluster(features_train_array,num_train_samples)






