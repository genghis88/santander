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
y = xTrain['TARGET']
zeroClass = xTrain['TARGET'] == 0
oneClass = xTrain['TARGET'] == 1
xTrain = xTrain.drop('ID', 1)

colsToBeRemoved = ['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var29_0', 'ind_var29', 'ind_var13_medio', 'ind_var18', 'ind_var26', 'ind_var25', 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'num_var29', 'num_var13_medio', 'num_var18', 'num_var26', 'num_var25', 'num_var32', 'num_var34', 'num_var37', 'num_var39', 'saldo_var29', 'saldo_medio_var13_medio_ult1', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3', 'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3']

for col in colsToBeRemoved:
  xTrain = xTrain.drop(col, 1)

xTrain = xTrain.fillna(0)
xTrain['var38'] = xTrain['var38'].apply(np.log)

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
  #trainX /= trainX.std(axis = None)
  #trainX -= trainX.mean()

  clf = GradientBoostingClassifier(loss='deviance', max_depth=5, n_estimators=350, learning_rate=0.03, subsample=0.95)
  scores = cross_validation.cross_val_score(rclf, trainX, trainY, cv=5)
  print('classifier ' + str(i) + ' ' + str(scores))

  clf.fit(trainX, trainY)
  classifiers.append(clf)'''

xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.as_matrix()
xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]).astype('float32')
y = y.as_matrix().astype('int32')
print(xTrain.shape)

#clf = GradientBoostingClassifier(loss='deviance', max_depth=5, n_estimators=350, learning_rate=0.03, subsample=0.95)
#scores = cross_validation.cross_val_score(clf, xTrain, y, cv=5)
#print(scores)
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
#scores = cross_validation.cross_val_score(clf, xTrain, y, cv=5, scoring='roc_auc')
#print(scores)

X_fit, X_eval, y_fit, y_eval= cross_validation.train_test_split(xTrain, y, test_size=0.3)
clf.fit(X_fit, y_fit, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_eval, y_eval)])
#clf.fit(xTrain, y, eval_metric='auc')

'''predictions = np.zeros((xTrain.shape[0], num_classifiers), dtype='int32')
for ind in range(num_classifiers):
  predictions[:,ind] = classifiers[ind].predict(xTrain)

def getSatisfaction(total, totalLimit):
  if total < totalLimit:
    return 0
  return 1

predictY = np.sum(predictions, axis=1)
satifactionFunc = np.vectorize(getSatisfaction)
predictY = satifactionFunc(predictY, num_classifiers/2)
print(np.count_nonzero(y - predictY) * 1.0 / xTrain.shape[0])'''

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  #pickle.dump(classifiers, f)
  pickle.dump(clf, f)
