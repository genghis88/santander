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

xTest = xTest.fillna(0)

xTest = xTest.as_matrix()

xTest = xTest.reshape(xTest.shape[0], xTest.shape[1]).astype('float32')
print(xTest.shape)

model = open(modelFile,'rb')

classifiers = pickle.load(model)
predictions = np.zeros((xTest.shape[0], len(classifiers)), dtype='int32')
for ind in range(len(classifiers)):
  predictions[:,ind] = classifiers[ind].predict_proba(xTest)[:,1]

def getSatisfaction(total, totalLimit):
  if total < totalLimit:
    return 0
  return 1

predictY = np.sum(predictions, axis=1) / len(classifiers)
#satifactionFunc = np.vectorize(getSatisfaction)
#predictY = satifactionFunc(predictY, len(classifiers)/2)

print(predictY)
print(predictY.shape)

#clf = pickle.load(model)
#predictY = clf.predict(xTest)
#predictY = clf.predict_proba(xTest)[:,1]
#print(predictY)

predictions = pd.DataFrame(index=samples)
predictions['TARGET'] = predictY
predictions.index.name = 'ID'
predictions.to_csv(predictionsFile)