import pandas as pd
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

trainFile = sys.argv[1]
testFile = sys.argv[2]
extractTrainFile = sys.argv[3]
extractTestFile = sys.argv[4]

xTrain = pd.read_csv(trainFile)
xTest = pd.read_csv(testFile)
y = xTrain['TARGET']
trainIds = xTrain['ID']
testIds = xTest['ID']
xTrain = xTrain.drop('ID', 1)
xTest = xTest.drop('ID', 1)

colsToBeRemoved = ['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var29_0', 'ind_var29', 'ind_var13_medio', 'ind_var18', 'ind_var26', 'ind_var25', 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'num_var29', 'num_var13_medio', 'num_var18', 'num_var26', 'num_var25', 'num_var32', 'num_var34', 'num_var37', 'num_var39', 'saldo_var29', 'saldo_medio_var13_medio_ult1', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3', 'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3']

for col in colsToBeRemoved:
  xTrain = xTrain.drop(col, 1)
  xTest = xTest.drop(col, 1)

xTrain = xTrain.fillna(0)
xTest = xTest.fillna(0)

batch_size = 100

xTrain = xTrain.drop('TARGET', 1)
xTrain['var38'] = xTrain['var38'].apply(np.log)
xTrain = xTrain.as_matrix()
xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]).astype('float32')
y = y.as_matrix().astype('int32')
xTest['var38'] = xTest['var38'].apply(np.log)
xTest = xTest.as_matrix()
xTest = xTest.reshape(xTest.shape[0], xTest.shape[1]).astype('float32')

'''xAll = np.concatenate((xTrain, xTest), axis=0)

print(xAll.shape)

pca = PCA(n_components=80)
pca.fit(xAll)
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.components_.shape)

transformedXTrain = pca.transform(xTrain)
scaler = StandardScaler()
scaler.fit(transformedXTrain)
transformedXTrain = scaler.transform(transformedXTrain)

transformedXTest = pca.transform(xTest)
transformedXTest = scaler.transform(transformedXTest)'''

#scaler = StandardScaler()
#scaler.fit(xTrain)
#transformedXTrain = scaler.transform(xTrain)
#transformedXTest = scaler.transform(xTest)
transformedXTrain = normalize(xTrain, axis=0)
transformedXTest = normalize(xTest, axis=0)

trainDF = pd.DataFrame(transformedXTrain, index=trainIds)
trainDF.index.name = 'ID'
trainDF['TARGET'] = y
testDF = pd.DataFrame(transformedXTest, index=testIds)
testDF.index.name = 'ID'

trainDF.to_csv(extractTrainFile)
testDF.to_csv(extractTestFile)
