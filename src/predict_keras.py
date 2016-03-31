import pickle
import sys
import pandas as pd
import keras.backend as K
import numpy as np

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
modelFile = sys.argv[2]
predictionsFile = sys.argv[3]

xTest = pd.read_csv(testFile)
samples = xTest['ID']
samples = samples.as_matrix()
xTest = xTest.drop('ID', 1)

xTest = xTest.fillna(0)

xTest = xTest.as_matrix()

#xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 1).astype('float32')
xTest = xTest.reshape(xTest.shape[0], xTest.shape[1]).astype('float32')
print(xTest.shape)

model = open(modelFile,'rb')

'''classifiers = pickle.load(model)
predictions = np.zeros((xTest.shape[0], len(classifiers)), dtype='int32')
for ind in range(len(classifiers)):
  predictions[:,ind] = classifiers[ind].predict(xTest)

def getSatisfaction(total, totalLimit):
  if total < totalLimit:
    return 0
  return 1

predictY = np.sum(predictions, axis=1)
satifactionFunc = np.vectorize(getSatisfaction)
predictY = satifactionFunc(predictY, len(classifiers)/2)

print(predictY)
print(predictY.shape)'''

clf = pickle.load(model)
#predictY = clf.predict(xTest)[:,0]
predictY = clf.predict_proba(xTest)[:,0]
print(predictY)
print(predictY.shape)

predictions = pd.DataFrame(index=samples)
predictions['TARGET'] = predictY
predictions.index.name = 'ID'
predictions.to_csv(predictionsFile)