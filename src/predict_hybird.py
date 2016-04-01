import pickle
import sys
import pandas as pd
import numpy as np

testFile = sys.argv[1]
model1File = sys.argv[2]
model2File = sys.argv[3]
predictionsFile = sys.argv[4]

xTest = pd.read_csv(testFile)
samples = xTest['ID']
samples = samples.as_matrix()
xTest = xTest.drop('ID', 1)

xTest = xTest.fillna(0)

xTest = xTest.as_matrix()

xTest = xTest.reshape(xTest.shape[0], xTest.shape[1]).astype('float32')
print(xTest.shape)

model1 = open(model1File,'rb')
model2 = open(model2File,'rb')

clf1 = pickle.load(model1)
predictY1 = clf1.predict_proba(xTest)[:,1]
print(predictY1)

clf2 = pickle.load(model2)
predictY2 = clf2.predict_proba(xTest)[:,0]
print(predictY2)

predictY = (predictY1 + predictY2) / 2

predictions = pd.DataFrame(index=samples)
predictions['TARGET'] = predictY
predictions.index.name = 'ID'
predictions.to_csv(predictionsFile)