import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv

data = pd.read_csv('./split_data/train_test_data.csv')
test_data = pd.read_csv('./same_season_test_data.csv')

def output_results(test_x, wlin, filename='results.csv'):

  results = np.dot(test_x, wlin)

  if not isinstance(results, np.ndarray):
    raise ValueError("Input 'results' must be a NumPy array.")

  signs = np.sign(results)  # Vectorized computation of signs

  with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Optional: Write a header row
    writer.writerow(['id', 'home_team_win'])
    
    # Write data rows
    for i, sign in enumerate(signs):
      if sign > 0:
        writer.writerow([i, True])
      else:
        writer.writerow([i, False])

# Preprocessing function
def preprocess_train_data(data):
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
  x = data.drop(columns=['home_team_win'], errors='ignore').filter(items = [
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

# Preprocessing function
def preprocess_test_data(data, features):
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
  x = data.filter(items = [
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

  # Normalize features
  scaler = StandardScaler()
  x = scaler.fit_transform(x)


  X_train = np.array([[example[i] for i in range(1, features+1)] for example in x])

  X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    
  return X_train_with_bias, label_encoders

def selectData(x, y, features, sampleSize):

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

def linearRegression(x, y, test_x, features, N):

  x_train, x_test, y_train, y_test = selectData(x, y, features, N)

  wlin = train(x_train, y_train)

  y_train_pred = np.dot(x_train, wlin) 

  Ein = np.sum(y_train_pred != y_train)

  y_test_pred = np.dot(x_test, wlin)
  
  accuracy = np.mean(np.sign(y_test_pred) == y_test)
  Eout = np.sum(y_test_pred != y_test)

  output_results(test_x, wlin)

  return Ein, Eout, accuracy

def preprocess_data(data, features):
  x, y, label_encoders = preprocess_train_data(data)
  test_x, label_encoders_2 = preprocess_test_data(test_data, features)

  return x, y, test_x

def plot_output(Ns, Eins, Eouts, accs):
  print('average accuracy:', np.mean(accs))
  print('max accuracy:', np.max(accs))
  plt.title('N VS Ein and Eout')
  plt.xlabel('N')
  plt.ylabel('Ein and Eout')

  # Show the plot
  plt.plot(Ns, Eins, color='red', alpha=0.4)
  plt.plot(Ns, Eouts, color='blue', alpha=0.4)
  plt.legend(['Ein', 'Eout'])
  plt.show()

  plt.title('N VS Accuracy')
  plt.xlabel('N')
  plt.ylabel('Accuracy')

  # Show the plot
  plt.plot(Ns, accs, color='green', alpha=0.4)
  plt.legend(['Accuracy'])
  plt.show()

def q10():
  acc = 0
  Eins, Eouts, accs = [], [], []
  features, sample_size = 7, 5000
  x, y, test_x = preprocess_data(data, features)
  
  while acc < 0.58:
    Ein, Eout, acc = linearRegression(x, y, test_x, features, sample_size)
    Eins.append(Ein)
    Eouts.append(Eout)
    accs.append(acc)
    
  print('average accuracy:', np.mean(accs))
  print('max accuracy:', np.max(accs))
  plt.title('Ein VS Eout')
  plt.xlabel('Ein')
  plt.ylabel('Eout')

  # Show the plot
  plt.scatter(Eins, Eouts, color='blue', alpha=0.4)

def q11():
  Ns, Eins, Eouts, accs = [], [], [], []
  features, N = 7, 25

  x, y, test_x = preprocess_data(data, features)

  while N < 9001:
    EinSum, EoutSum, accSum = 0, 0, 0

    for i in range(16):
      Ein, Eout, acc = linearRegression(x, y, test_x, features, N)
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

  plot_output(Ns, Eins, Eouts, accs)

def q12():
  features, Eins, Eouts, accs = [], [], [], []
  feature, N = 2, 5000

  while feature < 30:
    x, y, test_x = preprocess_data(data, feature)
    EinSum, EoutSum, accSum = 0, 0, 0

    for i in range(16):
      Ein, Eout, acc = linearRegression(x, y, test_x, feature, N)
      EinSum += Ein
      EoutSum += Eout
      accSum += acc

    EinSum /= 16
    EoutSum /= 16
    accSum /= 16

    features.append(feature)
    Eins.append(EinSum)
    Eouts.append(EoutSum)
    accs.append(accSum)

    feature = feature + 1

  plot_output(features, Eins, Eouts, accs)

q11()
