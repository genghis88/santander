import pickle
import sys
import pandas as pd
import numpy as np

testFile = sys.argv[1]
modelFile = sys.argv[2]
predictionsFile = sys.argv[3]

xTest = pd.read_csv(testFile)
samples = xTest['ID']
samples = samples.as_matrix()
xTest = xTest.drop('ID', 1)

colsToBeRemoved = ['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0', 'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var29_0', 'ind_var29', 'ind_var13_medio', 'ind_var18', 'ind_var26', 'ind_var25', 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'num_var29', 'num_var13_medio', 'num_var18', 'num_var26', 'num_var25', 'num_var32', 'num_var34', 'num_var37', 'num_var39', 'saldo_var29', 'saldo_medio_var13_medio_ult1', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3', 'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3']

for col in colsToBeRemoved:
  xTest = xTest.drop(col, 1)

xTest = xTest.fillna(0)
xTest['var38'] = xTest['var38'].apply(np.log)

xTest = xTest.as_matrix()

xTest = xTest.reshape(xTest.shape[0], xTest.shape[1]).astype('float32')
print(xTest.shape)

#xTest /= xTest.std(axis = None)
#xTest -= xTest.mean()

def writeToFile(samples, y, predictionsFile):
  with open(predictionsFile,'w') as output:
    output.write('ID,TARGET\n');
    for index, sample in np.ndenumerate(samples):
      prediction = y[index]
      output.write(str(sample) + ',' + str(prediction) + '\n')

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

rclf = pickle.load(model)
predictY = rclf.predict(xTest)

writeToFile(samples,predictY,predictionsFile)