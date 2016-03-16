import pandas as pd
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxoutDense, Activation
from keras.optimizers import SGD

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

print(pd.Series.value_counts(y))
xTrain = xTrain.drop('TARGET', 1)
xTrain = xTrain.as_matrix()
#xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1]).astype('float32')
xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]).astype('float32')
print(xTrain.shape)
#xTrain /= xTrain.std(axis = None)
#xTrain -= xTrain.mean()
y = y.as_matrix().astype('int32')
print(y.shape)
#y = map(ord, y)

net = Sequential()
net.add(Dense(160, input_dim=xTrain.shape[1], init='uniform', activation='relu'))
net.add(Dropout(0.5))
net.add(Dense(80, init='uniform', activation='tanh'))
net.add(Dropout(0.5))
net.add(Dense(40, init='uniform', activation='tanh'))
net.add(Dropout(0.5))
net.add(Dense(2, init='uniform', activation='softmax'))
net.add(MaxoutDense(1, init='uniform'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
net.compile(loss='binary_crossentropy',
              optimizer=sgd)

net.fit(xTrain, y,
          nb_epoch=100,
          batch_size=100,
          verbose=1,
          validation_split=0.2,
          show_accuracy=True)

#score = model.evaluate(X_test, y_test, batch_size=16)

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)