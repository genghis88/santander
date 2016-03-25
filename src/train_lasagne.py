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

colsToBeRemoved = ['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var29_0', 'ind_var29', 'ind_var13_medio', 'ind_var18', 'ind_var26', 'ind_var25', 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'num_var29', 'num_var13_medio', 'num_var18', 'num_var26', 'num_var25', 'num_var32', 'num_var34', 'num_var37', 'num_var39', 'saldo_var29', 'saldo_medio_var13_medio_ult1', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3', 'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3']

for col in colsToBeRemoved:
  xTrain = xTrain.drop(col, 1)

xTrain = xTrain.fillna(0)
xTrain['var38'] = xTrain['var38'].apply(np.log)

'''nets = []
num_classifiers = 100
for i in range(num_classifiers):
  xZeroTrain = xTrain.loc[zeroClass].sample(n=3008)
  xOneTrain = xTrain.loc[oneClass]
  trainX = xZeroTrain.append(xOneTrain)
  trainX = trainX.reindex(np.random.permutation(trainX.index)).fillna(0)
  #trainX = pd.concat([xZeroTrain, xOneTrain])
  trainY = trainX['TARGET']
  trainY = trainY.as_matrix().astype('int32')

  trainX = trainX.drop('TARGET', 1)
  trainX = trainX.as_matrix()
  trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1]).astype('float32')
  print(trainX.shape)
  #trainX /= trainX.std(axis = None)
  #trainX -= trainX.mean()

  inputLayer = layers.InputLayer(shape=(None, 1, trainX.shape[2]))

  #hidden1Layer = layers.DenseLayer(dropout6Layer, num_units=640, nonlinearity=elu)
  hidden1Layer = layers.DenseLayer(inputLayer, num_units=160, nonlinearity=elu)
  dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
  hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=80, nonlinearity=tanh)
  dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
  hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=40, nonlinearity=tanh)
  dropout3Layer = layers.DropoutLayer(hidden3Layer, p=0.5)
  hidden4Layer = layers.DenseLayer(dropout3Layer, num_units=20, nonlinearity=elu)
  dropout4Layer = layers.DropoutLayer(hidden4Layer, p=0.5)
  hidden5Layer = layers.DenseLayer(dropout4Layer, num_units=10, nonlinearity=tanh)
  hidden6Layer = layers.DenseLayer(hidden5Layer, num_units=5, nonlinearity=tanh)
  #hidden7Layer = layers.DenseLayer(hidden6Layer, num_units=10, nonlinearity=elu)
  #outputLayer = layers.DenseLayer(hidden6Layer, num_units=2, nonlinearity=sigmoid)
  outputLayer = layers.DenseLayer(hidden6Layer, num_units=2, nonlinearity=softmax)

  net = NeuralNet(
    layers = outputLayer,
    update_learning_rate = 0.01,
    #update_momentum = 0.9,

    batch_iterator_train = BatchIterator(batch_size = 100),
    batch_iterator_test = BatchIterator(batch_size = 100),

    use_label_encoder = True,
    #use_label_encoder = False,

    #objective=objectives,
    #objective_loss_function=objectives.binary_hinge_loss,

    regression = False,
    max_epochs = 200,
    verbose = 1
  )

  net.fit(trainX, trainY)
  nets.append(net)

predictions = np.zeros((xTrain.shape[0], num_classifiers), dtype='int32')
for ind in range(num_classifiers):
  predictions[:,ind] = nets[ind].predict(xTrain)

def getSatisfaction(total, totalLimit):
  if total < totalLimit:
    return 0
  return 1

predictY = np.sum(predictions, axis=1)
satifactionFunc = np.vectorize(getSatisfaction)
predictY = satifactionFunc(predictY, num_classifiers/2)
print(np.count_nonzero(y - predictY) * 1.0 / xTrain.shape[0])'''

'''conv1Layer = layers.Conv1DLayer(inputLayer, num_filters=10, filter_size=4)
conv2Layer = layers.Conv1DLayer(conv1Layer, num_filters=10, filter_size=4)
pool1Layer = layers.MaxPool1DLayer(conv2Layer, pool_size=2)
dropout1Layer = layers.DropoutLayer(pool1Layer, p=0.4)

conv3Layer = layers.Conv1DLayer(dropout1Layer, num_filters=20, filter_size=3)
pool2Layer = layers.MaxPool1DLayer(conv3Layer, pool_size=2)
dropout2Layer = layers.DropoutLayer(pool2Layer, p=0.4)

conv4Layer = layers.Conv1DLayer(dropout2Layer, num_filters=40, filter_size=3)
pool3Layer = layers.MaxPool1DLayer(conv4Layer, pool_size=2)
dropout3Layer = layers.DropoutLayer(pool3Layer, p=0.4)

conv5Layer = layers.Conv1DLayer(dropout3Layer, num_filters=80, filter_size=3)
pool4Layer = layers.MaxPool1DLayer(conv5Layer, pool_size=2)
dropout4Layer = layers.DropoutLayer(pool4Layer, p=0.4)

conv6Layer = layers.Conv1DLayer(dropout4Layer, num_filters=160, filter_size=4)
pool5Layer = layers.MaxPool1DLayer(conv6Layer, pool_size=2)
dropout5Layer = layers.DropoutLayer(pool5Layer, p=0.4)

conv7Layer = layers.Conv1DLayer(dropout5Layer, num_filters=320, filter_size=4)
pool6Layer = layers.MaxPool1DLayer(conv7Layer, pool_size=2)
dropout6Layer = layers.DropoutLayer(pool6Layer, p=0.4)'''

xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.as_matrix()
xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1]).astype('float32')
print(xTrain.shape)
y = y.as_matrix().astype('int32')

inputLayer = layers.InputLayer(shape=(None, 1, xTrain.shape[2]))
#inputLayer = layers.InputLayer(shape=(None, xTrain.shape[1]))
hidden1Layer = layers.DenseLayer(inputLayer, num_units=160, nonlinearity=elu)
dropout1Layer = layers.DropoutLayer(hidden1Layer, p=0.5)
hidden2Layer = layers.DenseLayer(dropout1Layer, num_units=80, nonlinearity=tanh)
dropout2Layer = layers.DropoutLayer(hidden2Layer, p=0.5)
hidden3Layer = layers.DenseLayer(dropout2Layer, num_units=40, nonlinearity=tanh)
#outputLayer = layers.DenseLayer(hidden3Layer, num_units=1, nonlinearity=elu)
outputLayer = layers.DenseLayer(hidden3Layer, num_units=2, nonlinearity=softmax)

net = NeuralNet(
  layers = outputLayer,
  update_learning_rate = 0.01,
  update_momentum = 0.9,

  batch_iterator_train = BatchIterator(batch_size = 100),
  batch_iterator_test = BatchIterator(batch_size = 100),

  use_label_encoder = True,
  #use_label_encoder = False,

  #objective=objectives,
  #objective_loss_function=objectives.binary_hinge_loss,

  regression = False,
  max_epochs = 100,
  verbose = 1
)

net.fit(xTrain, y)

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)