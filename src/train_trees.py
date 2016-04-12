import pandas as pd
import pickle
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxoutDense, Activation
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import cross_validation
import xgboost as xgb

trainFile = sys.argv[1]
pickleFile = sys.argv[2]

xTrain = pd.read_csv(trainFile)
xTrain = xTrain.reindex(np.random.permutation(xTrain.index))
y = xTrain['TARGET']
zeroClass = xTrain['TARGET'] == 0
oneClass = xTrain['TARGET'] == 1
xTrain = xTrain.drop('ID', 1)

xTrain = xTrain.fillna(0)

'''classifiers = []
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
  trainX = trainX.reshape(trainX.shape[0], trainX.shape[1]).astype('float32')
  print(trainX.shape)

  #clf = GradientBoostingClassifier(loss='deviance', max_depth=5, n_estimators=350, learning_rate=0.03, subsample=0.95)
  #clf = RandomForestClassifier(n_jobs=-1, n_estimators=350, max_features='auto')
  clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=500, learning_rate=0.01, nthread=4, subsample=0.95, colsample_bytree=0.85)
  scores = cross_validation.cross_val_score(clf, trainX, trainY, cv=5, scoring='roc_auc')
  print('classifier ' + str(i) + ' ' + str(scores))

  clf.fit(trainX, trainY)
  classifiers.append(clf)'''

xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.as_matrix()
xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]).astype('float32')
y = y.as_matrix().astype('int32')
print(xTrain.shape)

#clf = GradientBoostingClassifier(loss='deviance', max_depth=5, n_estimators=350, learning_rate=0.03, subsample=0.95)
#clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, criterion='entropy', max_features=None, min_samples_leaf=100, min_weight_fraction_leaf=0.01, random_state=np.random.RandomState(seed=1352), class_weight='balanced_subsample')
#clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1, n_estimators=50, criterion='entropy', max_features=None, min_samples_leaf=100, min_weight_fraction_leaf=0.01, random_state=np.random.RandomState(seed=1352), class_weight='balanced_subsample'), n_estimators=100, learning_rate=0.01, algorithm='SAMME.R')
clf = AdaBoostClassifier(base_estimator=xgb.XGBClassifier(objective='binary:logistic', n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=2471), n_estimators=100, learning_rate=0.01, algorithm='SAMME.R')
scores = cross_validation.cross_val_score(clf, xTrain, y, cv=5, scoring='roc_auc')
print(scores)
clf.fit(xTrain, y)
#clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=2471)
#scores = cross_validation.cross_val_score(clf, xTrain, y, cv=5, scoring='roc_auc')
#print(scores)

#X_fit, X_eval, y_fit, y_eval= cross_validation.train_test_split(xTrain, y, test_size=0.3)
#clf.fit(X_fit, y_fit, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_eval, y_eval)])
#clf.fit(xTrain, y, eval_metric='auc')

predictY = clf.predict_proba(xTrain)[:,1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_true = le.fit_transform(y)
print(y_true)
from sklearn.metrics import roc_auc_score
print('SCORE 1 ' + str(roc_auc_score(y_true, predictY)))

#from sklearn.preprocessing import MinMaxScaler
#minMaxScaler = MinMaxScaler()
#predictY = minMaxScaler.fit_transform(predictY)

#print('SCORE 2 ' + str(roc_auc_score(y, predictY)))

'''predictions = np.zeros((xTrain.shape[0], num_classifiers), dtype='int32')
for ind in range(num_classifiers):
  predictionY = classifiers[ind].predict_proba(xTrain)[:,1]
  print(predictionY)
  predictions[:,ind] = predictionY

def getSatisfaction(total, totalLimit):
  if total < totalLimit:
    return 0
  return 1

#predictY = np.sum(predictions, axis=1) / (num_classifiers * 1.0)
predictY = np.average(predictions, axis=1)
print(predictY)
#satifactionFunc = np.vectorize(getSatisfaction)
#predictY = satifactionFunc(predictY, num_classifiers/2)
#print(np.count_nonzero(y - predictY) * 1.0 / xTrain.shape[0])

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y, predictY))'''

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  #pickle.dump(classifiers, f)
  pickle.dump(clf, f)
