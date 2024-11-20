import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data-for-stage-1/train_data_copy_jason.csv')

data = data

def selectData(features, sampleSize):

  X_train = np.array([[example[i] for i in range(1, features+1)] for example in data.to_numpy()])

  X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

  x_train, x_test, y_train, y_test = train_test_split(X_train_with_bias, y, train_size=sampleSize, test_size=8192-sampleSize)

  return x_train, x_test, y_train, y_test

def train(x, y):

  X_transpose = x.T

  XTX = np.dot(X_transpose, x) 

  XTX_inv = np.linalg.inv(XTX) 

  XTX_inv_XT = np.dot(XTX_inv, X_transpose)

  wlin = np.dot(XTX_inv_XT, y) 

  return wlin

def linearRegression(features, N):

  x_train, x_test, y_train, y_test = selectData(features, N)

  wlin = train(x_train, y_train)

  y_train_pred = np.dot(x_train, wlin) 

  Ein = np.mean((y_train_pred - y_train) ** 2)

  y_test_pred = np.dot(x_test, wlin)
  
  Eout = np.mean((y_test_pred - y_test) ** 2)

  return Ein, Eout

def q10():
  Eins, Eouts = [], []
  for i in range(1126):
    Ein, Eout = linearRegression(12, 32)
    Eins.append(Ein)
    Eouts.append(Eout)

  plt.title('Ein VS Eout')
  plt.xlabel('Ein')
  plt.ylabel('Eout')

  # Show the plot
  plt.scatter(Eins, Eouts, color='blue', alpha=0.4)
  plt.show()

def q11():
  N = 25
  Ns = []
  Eins = []
  Eouts = []

  while N < 2001:
    EinSum = 0
    EoutSum = 0

    for i in range(16):
      Ein, Eout = linearRegression(12, N)
      EinSum += Ein
      EoutSum += Eout

    EinSum /= 16
    EoutSum /= 16

    Ns.append(N)
    Eins.append(EinSum)
    Eouts.append(EoutSum)

    N = N + 25

  plt.title('N VS Ein and Eout')
  plt.xlabel('N')
  plt.ylabel('Ein and Eout')

  # Show the plot
  plt.plot(Ns, Eins, color='red', alpha=0.4)
  plt.plot(Ns, Eouts, color='blue', alpha=0.4)
  plt.legend(['Ein', 'Eout'])
  plt.show()

def q12():
  N = 25
  Ns = []
  Eins = []
  Eouts = []

  while N < 2001:
    EinSum = 0
    EoutSum = 0

    for i in range(16):
      Ein, Eout = linearRegression(2, N)
      EinSum += Ein
      EoutSum += Eout

    EinSum /= 16
    EoutSum /= 16

    Ns.append(N)
    Eins.append(EinSum)
    Eouts.append(EoutSum)

    N = N + 25

  plt.title('N VS Ein and Eout')
  plt.xlabel('N')
  plt.ylabel('Ein and Eout')

  # Show the plot
  plt.plot(Ns, Eins, color='red', alpha=0.4)
  plt.plot(Ns, Eouts, color='blue', alpha=0.4)
  plt.legend(['Ein', 'Eout'])
  plt.show()

q12()