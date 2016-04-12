import pickle
import sys
import pandas as pd
import numpy as np
import keras.backend as K

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

testFile = sys.argv[1]
trainFile = sys.argv[2]
model1File = sys.argv[3]
model2File = sys.argv[4]
model3File = sys.argv[5]
predictionsFile = sys.argv[6]

xTest = pd.read_csv(testFile)
samples = xTest['ID']
samples = samples.as_matrix()
xTest = xTest.drop('ID', 1)

xTest = xTest.fillna(0)

xTest = xTest.as_matrix()

xTest = xTest.reshape(xTest.shape[0], xTest.shape[1]).astype('float32')
print(xTest.shape)

xTrain = pd.read_csv(trainFile)
samplesTrain = xTrain['ID']
samplesTrain = samplesTrain.as_matrix()
xTrain = xTrain.drop('ID', 1)
y = xTrain['TARGET']
xTrain = xTrain.drop('TARGET', 1)

y = y.as_matrix().astype('int32')

xTrain = xTrain.fillna(0)

xTrain = xTrain.as_matrix()

xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]).astype('float32')
print(xTrain.shape)

model1 = open(model1File,'rb')
model2 = open(model2File,'rb')
model3 = open(model3File,'rb')

clf1 = pickle.load(model1)
predictY1 = clf1.predict_proba(xTest)[:,1]
trainingPredictY1 = clf1.predict_proba(xTrain)[:,1]
print(predictY1)

clf2 = pickle.load(model2)
predictY2 = clf2.predict_proba(xTest)[:,1]
trainingPredictY2 = clf2.predict_proba(xTrain)[:,1]
print(predictY2)

clf3 = pickle.load(model3)
#xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], 1).astype('float32')
#xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 1).astype('float32')
predictY3 = clf3.predict_proba(xTest)[:,0]
trainingPredictY3 = clf3.predict_proba(xTrain)[:,0]
print(predictY3)

predictY = (1 * predictY1 + 1 * predictY2 + 0 * predictY3) / 2

trainingPredictY = (1 * trainingPredictY1 + 1 * trainingPredictY2 + 0 * trainingPredictY3) / 2
#trainingPredictY = trainingPredictY3

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y, trainingPredictY))

predictions = pd.DataFrame(index=samples)
predictions['TARGET'] = predictY
predictions.index.name = 'ID'
predictions.to_csv(predictionsFile)