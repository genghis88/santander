import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

predictionsFlie = sys.argv[1]
preds = pd.read_csv(predictionsFlie)

predictions = preds['TARGET']

#plt.hist(predictions, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

def rescale(val):
  if val <= 0.5:
    return val/10
  else:
    return 1.0 - (1.0 - val) / 10
  '''elif val <= 0.5:
    return val
  elif val <= 0.75:
    return val'''

predictions = predictions.apply(rescale)

print(predictions)

#preds['TARGET'] = predictions

#preds.to_csv('predictions/test.csv', index=False)

plt.hist(predictions, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.show()