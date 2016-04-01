import pandas as pd
import pickle
import sys
import numpy as np
import keras
import keras.backend as K
import classification
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxoutDense, Activation, Convolution1D, MaxPooling1D, Flatten, RepeatVector
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Adamax
from keras.layers.advanced_activations import ELU, PReLU
from sklearn import metrics

import theano.tensor as T

def binary_crossentropy_with_ranking(y_true, y_pred):
  """ Trying to combine ranking loss with numeric precision"""
  # first get the log loss like normal
  logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

  # next, build a rank loss

  # clip the probabilities to keep stability
  y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

  # translate into the raw scores before the logit
  y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

  # determine what the maximum score for a zero outcome is
  y_pred_score_zerooutcome_max = K.max(y_pred_score * (y_true <1))

  # determine how much each score is above or below it
  rankloss = y_pred_score - y_pred_score_zerooutcome_max

  # only keep losses for positive outcomes
  rankloss = rankloss * y_true

  # only keep losses where the score is below the max
  rankloss = K.square(K.clip(rankloss, -100, 0))

  # average the loss for just the positive outcomes
  rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)

  # return (rankloss + 1) * logloss - an alternative to try
  return rankloss + logloss

_EPSILON = 1.0e-8

def custom_obj(target, output):
  output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
  return T.mean(T.nnet.binary_crossentropy(output, target), axis=-1, keepdims=False, dtype=None)

def custom_objective(target, output):
  indices0 = np.where(target == 0)
  indices1 = np.where(target != 0)
  print('indices ' + str(indices0))
  return (custom_obj(target[indices0], output[indices0]) + custom_obj(target[indices1], output[indices1])) / 2.0

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

xTrain = pd.read_csv(trainFile)
y = xTrain['TARGET']
zeroClass = xTrain['TARGET'] == 0
oneClass = xTrain['TARGET'] == 1
xTrain = xTrain.drop('ID', 1)

xTrain = xTrain.fillna(0)

batch_size = 100

'''nets = []
num_classifiers = 10
for i in range(num_classifiers):
  xZeroTrain = xTrain.loc[zeroClass].sample(n=3008)
  xOneTrain = xTrain.loc[oneClass]
  trainX = xZeroTrain.append(xOneTrain)
  trainX = trainX.reindex(np.random.permutation(trainX.index)).fillna(0)
  #trainX = trainX.fillna(0)
  #trainX = pd.concat([xZeroTrain, xOneTrain])
  trainY = trainX['TARGET']
  trainY = trainY.as_matrix().astype('int32')

  trainX = trainX.drop('TARGET', 1)
  trainX = trainX.as_matrix()
  trainX = trainX.reshape(trainX.shape[0], trainX.shape[1]).astype('float32')
  print(trainX.shape)
  #trainX /= trainX.std(axis = None)
  #trainX -= trainX.mean()

  net = Sequential()
  net.add(Dense(160, input_dim=trainX.shape[1], init='uniform', activation='relu'))
  net.add(Dropout(0.5))
  net.add(Dense(80, init='uniform', activation='tanh'))
  net.add(Dropout(0.5))
  net.add(Dense(40, init='uniform', activation='tanh'))
  net.add(Dropout(0.5))
  net.add(Dense(20, init='uniform', activation='tanh'))
  #net.add(Dense(2, init='uniform', activation='softmax'))
  net.add(Dense(1, init='uniform', activation='sigmoid'))

  optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
  net.compile(loss='binary_crossentropy',
              optimizer=optimizer, class_mode='binary')

  net.fit(trainX, trainY,
          nb_epoch=40,
          batch_size=100,
          verbose=1,
          validation_split=0.2,
          show_accuracy=True)

  nets.append(net)'''


print(pd.Series.value_counts(y))
xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.as_matrix()
#xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], 1).astype('float32')
xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]).astype('float32')
print(xTrain.shape)
y = y.as_matrix().astype('int32')
print(y.shape)

'''net = Sequential()
net.add(Convolution1D(nb_filter=100, filter_length=3, border_mode='valid', input_shape=(xTrain.shape[1],1), init='uniform'))
net.add(ELU())
net.add(MaxPooling1D(2))
net.add(Dropout(0.5))
#net.add(Convolution1D(50, 5, init='uniform', activation='tanh'))
#net.add(MaxPooling1D(2))
#net.add(Dropout(0.5))
#net.add(Convolution1D(80, 5, init='uniform'))
#net.add(MaxPooling1D(2))
#net.add(Dropout(0.5))
net.add(Flatten())
net.add(Dense(200, init='uniform', activation='tanh'))
#net.add(ELU())
net.add(Dropout(0.5))
net.add(Dense(100, init='uniform', activation='tanh'))
net.add(Dropout(0.5))
net.add(Dense(50, init='uniform'))
net.add(ELU())
net.add(Dropout(0.5))
net.add(Dense(25, init='uniform', activation='tanh'))
#net.add(ELU())
#net.add(Dropout(0.5))
#net.add(Dense(5, init='uniform', activation='tanh'))
#net.add(Dropout(0.5))
#net.add(Dense(40, init='uniform', activation='tanh'))
#net.add(Dense(2, init='uniform', activation='softmax'))
net.add(Dense(1, init='uniform', activation='sigmoid'))'''

net = Sequential()
net.add(Dense(320, input_dim=xTrain.shape[1], init='glorot_uniform'))
net.add(ELU())
net.add(Dropout(0.5))
net.add(Dense(160, init='glorot_uniform', activation='sigmoid'))
net.add(Dropout(0.5))
net.add(Dense(80, init='glorot_uniform'))
net.add(ELU())
net.add(Dropout(0.5))
net.add(Dense(40, init='glorot_uniform', activation='softplus'))
'''#net.add(Dropout(0.5))
net.add(Dense(20, init='uniform'))
net.add(ELU())
net.add(Dropout(0.5))
net.add(Dense(10, init='uniform', activation='tanh'))
#net.add(Dense(2, init='uniform', activation='softmax'))'''
net.add(Dense(1, init='uniform', activation='sigmoid'))

#optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10)
#optimizer = Adagrad(lr=0.01, epsilon=1e-06)
#optimizer =  Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#net.compile(loss=binary_crossentropy_with_ranking,
#              optimizer=optimizer, class_mode='binary')
net.compile(loss='binary_crossentropy',
               optimizer=optimizer, class_mode='binary')
#net.compile(loss='hinge',
#               optimizer=optimizer, class_mode='binary')

net.fit(xTrain, y,
          nb_epoch=30,
          batch_size=batch_size,
          verbose=1,
          validation_split=0.2,
          show_accuracy=True,
          #shuffle=True,
          class_weight={0:0.0396, 1:0.9604})

#score = model.evaluate(X_test, y_test, batch_size=16)

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)