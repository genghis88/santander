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
from lasagne import objectives
from nolearn.lasagne import objective
from lasagne.init import Constant, Uniform, GlorotUniform
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

xTrain = pd.read_csv(trainFile)
y = xTrain['TARGET']
zeroClass = xTrain['TARGET'] == 0
oneClass = xTrain['TARGET'] == 1
xTrain = xTrain.drop('ID', 1)

xTrain = xTrain.fillna(0)

xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.as_matrix()
xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1]).astype('float32')
print(xTrain.shape)
y = y.as_matrix().astype('int32')

inputLayer = layers.InputLayer(shape=(None, 1, xTrain.shape[2]))
#inputLayer = layers.InputLayer(shape=(None, xTrain.shape[1]))
hidden1Layer = layers.DenseLayer(inputLayer, num_units=320, nonlinearity=elu)
dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=160, nonlinearity=tanh)
dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=80, nonlinearity=tanh)
dropout3Layer = layers.DropoutLayer(hidden3Layer, p=0.5)
hidden4Layer = layers.DenseLayer(dropout3Layer, num_units=40, nonlinearity=elu)
dropout4Layer = layers.DropoutLayer(hidden4Layer, p=0.5)
hidden5Layer = layers.DenseLayer(dropout4Layer, num_units=20, nonlinearity=tanh)
#outputLayer = layers.DenseLayer(hidden3Layer, num_units=1, nonlinearity=elu)
#outputLayer = layers.DenseLayer(hidden3Layer, num_units=2, nonlinearity=softmax)
outputLayer = layers.DenseLayer(hidden5Layer, num_units=1, nonlinearity=sigmoid)

net = NeuralNet(
  layers = outputLayer,
  update_learning_rate = 0.01,
  update_momentum = 0.9,

  batch_iterator_train = BatchIterator(batch_size = 100),
  batch_iterator_test = BatchIterator(batch_size = 100),

  use_label_encoder = True,
  #use_label_encoder = False,

  #objective=objectives,
  objective_loss_function=objectives.binary_crossentropy,

  regression = False,
  max_epochs = 100,
  verbose = 1
)

net.fit(xTrain, y)

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)