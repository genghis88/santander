import pandas as pd
from PIL import Image
import pickle
from os.path import join
import sys
import numpy as np
from lasagne import layers
from lasagne.init import Constant
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh, ScaledTanH, elu, identity, softplus, leaky_rectify
from nolearn.lasagne import NeuralNet, BatchIterator

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

xTrain = pd.read_csv(trainFile)
y = xTrain['TARGET']
xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.drop('ID', 1)
xTrain = xTrain.as_matrix()

xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1]).astype('float32')
print(xTrain.shape)

xTrain /= xTrain.std(axis = None)
xTrain -= xTrain.mean()

y = y.as_matrix()
print(y.shape)
#y = map(ord, y)

inputLayer = layers.InputLayer(shape=(None, 1, xTrain.shape[2]))

#Classification network
conv1Layer = layers.Conv1DLayer(inputLayer, num_filters=32, filter_size=4)
pool1Layer = layers.MaxPool1DLayer(conv1Layer, pool_size=2)
dropout1Layer = layers.DropoutLayer(pool1Layer, p=0.2)

conv2Layer = layers.Conv1DLayer(dropout1Layer, num_filters=64, filter_size=4)
pool2Layer = layers.MaxPool1DLayer(conv2Layer, pool_size=2)
dropout2Layer = layers.DropoutLayer(pool2Layer, p=0.2)

conv3Layer = layers.Conv1DLayer(dropout2Layer, num_filters=128, filter_size=5)
pool3Layer = layers.MaxPool1DLayer(conv3Layer, pool_size=2)
dropout3Layer = layers.DropoutLayer(pool3Layer, p=0.2)

conv4Layer = layers.Conv1DLayer(dropout3Layer, num_filters=256, filter_size=4)
pool4Layer = layers.MaxPool1DLayer(conv4Layer, pool_size=2)
dropout4Layer = layers.DropoutLayer(pool4Layer, p=0.2)

conv5Layer = layers.Conv1DLayer(dropout4Layer, num_filters=512, filter_size=5)

hidden1Layer = layers.DenseLayer(conv5Layer, num_units=8192, nonlinearity=elu)
hidden2Layer = layers.DenseLayer(hidden1Layer, num_units=4096, nonlinearity=tanh)
hidden3Layer = layers.DenseLayer(hidden2Layer, num_units=2048, nonlinearity=tanh)
hidden4Layer = layers.DenseLayer(hidden3Layer, num_units=1024, nonlinearity=elu)
hidden5Layer = layers.DenseLayer(hidden4Layer, num_units=512, nonlinearity=tanh)
hidden6Layer = layers.DenseLayer(hidden5Layer, num_units=256, nonlinearity=tanh)
hidden7Layer = layers.DenseLayer(hidden6Layer, num_units=128, nonlinearity=elu)
hidden8Layer = layers.DenseLayer(hidden7Layer, num_units=64, nonlinearity=tanh)
hidden9Layer = layers.DenseLayer(hidden8Layer, num_units=32, nonlinearity=tanh)
outputLayer = layers.DenseLayer(hidden8Layer, num_units=2, nonlinearity=softmax)

net = NeuralNet(
  layers = outputLayer,
  update_learning_rate = 0.01,
  update_momentum = 0.9,
  
  batch_iterator_train = BatchIterator(batch_size = 100),
  batch_iterator_test = BatchIterator(batch_size = 100),
  
  use_label_encoder = True,
  #use_label_encoder = False,
  regression = False,
  max_epochs = 2,
  verbose = 1
)

net.fit(xTrain, y)

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)
