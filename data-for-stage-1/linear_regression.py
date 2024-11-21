import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('./train_data.csv')

# Preprocessing function
def preprocess_data(data):
  # Drop non-predictive columns
  drop_cols = ['id', 'date']
  data = data.drop(columns=drop_cols, errors='ignore')
  
  # Handle categorical variables
  categorical_cols = data.select_dtypes(include=['object']).columns
  label_encoders = {}
  for col in categorical_cols:
      le = LabelEncoder()
      data[col] = le.fit_transform(data[col].astype(str))
      label_encoders[col] = le  # Save encoders if needed later
  
  # Handle missing values
  data = data.fillna(data.median(numeric_only=True))

  # Split features and target
  x = data.drop(columns=['home_team_win']).filter(items = [
    "away_pitching_SO_batters_faced_10RA",
    "home_pitching_earned_run_avg_mean",
    "away_pitcher_SO_batters_faced_mean",
    "home_batting_onbase_plus_slugging_mean",
    "home_pitching_H_batters_faced_10RA",
    "home_pitching_SO_batters_faced_10RA",
    "away_pitching_SO_batters_faced_mean",
    "home_pitching_BB_batters_faced_mean",
    "home_batting_onbase_plus_slugging_skew",
    "away_pitching_leverage_index_avg_mean",
    "away_pitching_SO_batters_faced_std",
    "home_batting_onbase_plus_slugging_std",
    "away_team_spread_mean",
    "away_pitching_wpa_def_skew",
    "away_pitching_BB_batters_faced_mean",
    "away_batting_onbase_perc_mean",
    "home_batting_onbase_perc_mean",
    "home_batting_batting_avg_10RA",
    "home_pitching_earned_run_avg_10RA",
    "home_pitching_wpa_def_skew",
    "away_pitching_earned_run_avg_10RA",
    "home_pitching_H_batters_faced_mean",
    "away_pitching_H_batters_faced_std",
    "away_batting_leverage_index_avg_10RA",
    "home_pitching_BB_batters_faced_std",
    "away_batting_leverage_index_avg_skew",
    "away_pitcher_earned_run_avg_mean",
    "away_pitching_wpa_def_mean",
    "away_pitcher_SO_batters_faced_10RA",
    "away_batting_onbase_plus_slugging_mean"
    ]).to_numpy()
  y = data['home_team_win'].map({True: 1, False: -1}).to_numpy()

  # Normalize features
  scaler = StandardScaler()
  x = scaler.fit_transform(x)

  print('Data shape:', x.shape, y.shape)
    
  return x, y, label_encoders

x, y, label_encoders = preprocess_data(data)


def selectData(features, sampleSize):

  X_train = np.array([[example[i] for i in range(1, features+1)] for example in x])

  X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

  x_train, x_test, y_train, y_test = train_test_split(X_train_with_bias, y, train_size=sampleSize, test_size=y.shape[0]-sampleSize)

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
  
  accuracy = np.mean(np.sign(y_test_pred) == y_test)
  Eout = np.mean((y_test_pred - y_test) ** 2)

  return Ein, Eout, accuracy

def q10():
  Eins, Eouts, accs = [], [], []
  for i in range(1126):
    Ein, Eout, acc = linearRegression(16, 1000)
    Eins.append(Ein)
    Eouts.append(Eout)
    accs.append(acc)
    print(acc)
    
  print('average accuracy:', np.mean(accs))
  print('max accuracy:', np.max(accs))
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
  accs = []

  while N < 10001:
    EinSum = 0
    EoutSum = 0
    accSum = 0

    for i in range(16):
      Ein, Eout, acc = linearRegression(163, N)
      EinSum += Ein
      EoutSum += Eout
      accSum += acc

    EinSum /= 16
    EoutSum /= 16
    accSum /= 16

    Ns.append(N)
    Eins.append(EinSum)
    Eouts.append(EoutSum)
    accs.append(accSum)

    N = N + 25

  plt.title('N VS Ein and Eout')
  plt.xlabel('N')
  plt.ylabel('Ein and Eout')

  # Show the plot
  plt.plot(Ns, Eins, color='red', alpha=0.4)
  plt.plot(Ns, Eouts, color='blue', alpha=0.4)
  plt.plot(Ns, accs, color='green', alpha=0.4)
  plt.legend(['Ein', 'Eout', 'Accuracy'])
  plt.show()

def q12():
  N = 25
  Ns = []
  Eins = []
  Eouts = []
  accs = []

  while N < 10001:
    EinSum = 0
    EoutSum = 0
    accSum = 0

    for i in range(16):
      Ein, Eout, acc = linearRegression(2, N)
      EinSum += Ein
      EoutSum += Eout
      accSum += acc

    EinSum /= 16
    EoutSum /= 16
    accSum /= 16

    Ns.append(N)
    Eins.append(EinSum)
    Eouts.append(EoutSum)
    accs.append(accSum)

    N = N + 25

  plt.title('N VS Ein and Eout')
  plt.xlabel('N')
  plt.ylabel('Ein and Eout')

  # Show the plot
  plt.plot(Ns, Eins, color='red', alpha=0.4)
  plt.plot(Ns, Eouts, color='blue', alpha=0.4)
  plt.plot(Ns, accs, color='green', alpha=0.4)
  plt.legend(['Ein', 'Eout', 'Accuracy'])
  plt.show()

q10()
