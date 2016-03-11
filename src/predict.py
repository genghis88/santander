import pickle
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np

testFile = sys.argv[1]
modelFile = sys.argv[2]
predictionsFile = sys.argv[3]

xTest = pd.read_csv(testFile)
xTest = xTest.drop('TARGET', 1)
samples = xTest['ID']
samples = samples.as_matrix()
xTest = xTest.drop('ID', 1)
xTest = xTest.as_matrix()

xTest = xTest.reshape(xTest.shape[0], 1, xTest.shape[1]).astype('float32')
print(xTest.shape)

xTest /= xTest.std(axis = None)
xTest -= xTest.mean()

def writeToFile(samples, y, predictionsFile):
  with open(predictionsFile,'w') as output:
    output.write('ID,TARGET\n');
    for index, sample in np.ndenumerate(samples):
      prediction = y[index]
      output.write(sample + ',' + prediction + '\n')

model = open(modelFile,'rb')
net = pickle.load(model)
y = net.predict(xTest)
print y
writeToFile(samples,y,predictionsFile)
