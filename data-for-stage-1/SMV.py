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
  x = data.drop(columns=['home_team_win']).to_numpy()
  y = data['home_team_win'].map({True: 1, False: -1}).to_numpy()

  # Normalize features
  scaler = StandardScaler()
  x = scaler.fit_transform(x)

  print('Data shape:', x.shape, y.shape)
    
  return x, y, label_encoders

x, y, label_encoders = preprocess_data(data)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def selectData(features, sampleSize):
    X_train = np.array([[example[i] for i in range(1, features+1)] for example in x])
    X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    x_train, x_test, y_train, y_test = train_test_split(X_train_with_bias, y, train_size=sampleSize, test_size=y.shape[0]-sampleSize)
    return x_train, x_test, y_train, y_test

def selectPolyData(features, sampleSize, Q):
    X_train = np.array([[example[i] for i in range(1, features+1)] for example in x])
    transformed = X_train.copy()
    for i in range(1, Q):
        poly_features = np.power(X_train, i)
        transformed = np.hstack([transformed, poly_features])
    X_train_with_bias = np.hstack([np.ones((transformed.shape[0], 1)), transformed])
    x_train, x_test, y_train, y_test = train_test_split(X_train_with_bias, y, train_size=sampleSize, test_size=y.shape[0]-sampleSize)
    return x_train, x_test, y_train, y_test

def compute_error(w, X, y):
    predictions = X.dot(w)
    return np.mean((predictions - y) ** 2), np.mean(np.sign(predictions) != y)

def gradientDescent(iterations, eta, x_train, x_test, y_train, y_test):
    # Initialize weights
    w = np.zeros(x_train.shape[1])
    interval = 200
    Ein_record, Eout_record = [], []
    Ein_accs, Eout_accs = [], []

    for t in range(1, iterations + 1):
        # Pick a random training example
        i = np.random.randint(len(x_train))
        xi = x_train[i]
        yi = y_train[i]
        
        # Compute gradient for the selected example
        gradient = 2 * (np.dot(w, xi) - yi) * xi
        
        # Update weights
        w -= eta * gradient

        # Record Ein and Eout every 200 iterations
        if t % interval == 0:
            Ein, Ein_acc = compute_error(w, x_train, y_train)
            Eout, Eout_acc = compute_error(w, x_test, y_test)
            Ein_record.append(Ein)
            Eout_record.append(Eout)
            Ein_accs.append(Ein_acc)
            Eout_accs.append(Eout_acc)


    return w, Ein_record, Eout_record, Ein_accs, Eout_accs

def train(x, y):
    X_transpose = x.T
    XTX = np.dot(X_transpose, x)
    XTX_inv = np.linalg.pinv(XTX)
    XTX_inv_XT = np.dot(XTX_inv, X_transpose)
    wlin = np.dot(XTX_inv_XT, y)
    return wlin

def linearRegression(x_train, x_test, y_train, y_test):
    wlin = train(x_train, y_train)
    y_train_pred = np.dot(x_train, wlin)
    Ein = np.mean((y_train_pred - y_train) ** 2)
    y_test_pred = np.dot(x_test, wlin)
    accuracy = np.mean(np.sign(y_test_pred) == y_test)
    Eout = np.mean((y_test_pred - y_test) ** 2)
    return Ein, Eout, accuracy

def plot_errors(Ein_avg, Eout_avg, avg_Ein_wlin, avg_Eout_wlin, interval, iterations):
    t_values = np.arange(interval, iterations + 1, interval)
    plt.plot(t_values, Ein_avg, label="Average Ein(wt)", color="blue")
    plt.plot(t_values, Eout_avg, label="Average Eout(wt)", color="red")
    plt.axhline(y=avg_Ein_wlin, color="blue", linestyle="--", label="Average Ein(wlin)")
    plt.axhline(y=avg_Eout_wlin, color="red", linestyle="--", label="Average Eout(wlin)")
    plt.xlabel("Iterations (t)")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Average Ein and Eout over Iterations (SGD for Linear Regression)")
    plt.show()

def q10():
    N = 1000
    numDim = 12
    eta = 0.01
    iterations = 100000
    interval = 200

    Eins_wlin, Eouts_wlin, accs  = [], [], []
    Eins_wt, Eouts_wt, Ein_accs, Eout_accs = [], [], [], []
    for i in range(1126):
        print("run number ", i)
        x_train, x_test, y_train, y_test = selectData(numDim, N)
        Ein, Eout, acc = linearRegression(x_train, x_test, y_train, y_test)
        _, Ein_record, Eout_record, Ein_acc, Eout_acc = gradientDescent(iterations, eta, x_train, x_test, y_train, y_test)
        Eins_wlin.append(Ein)
        Eouts_wlin.append(Eout)
        accs.append(acc)
        Eins_wt.append(Ein_record)
        Eouts_wt.append(Eout_record)
        Ein_accs.append(Ein_acc)
        Eout_accs.append(Eout_acc)

    # Calculate average errors at each interval
    avg_Ein_wlin = np.mean(Eins_wlin)
    avg_Eout_wlin = np.mean(Eouts_wlin)
    avg_acc = np.mean(accs)
    Ein_avg = np.mean(Eins_wt, axis=0)
    Eout_avg = np.mean(Eouts_wt, axis=0)
    avg_Ein_acc = np.mean(Ein_accs)
    avg_Eout_acc = np.mean(Eout_accs)

    # Plot errors
    plot_errors(Ein_avg, Eout_avg, avg_Ein_wlin, avg_Eout_wlin, interval, iterations)
    print('average accuracy:', avg_acc)
    print('average Ein accuracy:', avg_Ein_acc)
    print('average Eout accuracy:', avg_Eout_acc)

    return Eins_wlin, Eouts_wlin

def q11():
    N = 64
    Q = 3
    numDim = 12
    eta = 0.01
    iterations = 100000
    interval = 200

    Eins_wlin, Eouts_wlin, accs  = [], [], []
    Eins_wt, Eouts_wt, Ein_accs, Eout_accs = [], [], [], []
    for i in range(1126):
        print("run number ", i)
        x_train, x_test, y_train, y_test = selectPolyData(numDim, N, Q)
        Ein, Eout, acc = linearRegression(x_train, x_test, y_train, y_test)
        _, Ein_record, Eout_record, Ein_acc, Eout_acc = gradientDescent(iterations, eta, x_train, x_test, y_train, y_test)
        Eins_wlin.append(Ein)
        Eouts_wlin.append(Eout)
        accs.append(acc)
        Eins_wt.append(Ein_record)
        Eouts_wt.append(Eout_record)
        Ein_accs.append(Ein_acc)
        Eout_accs.append(Eout_acc)

    # Calculate average errors at each interval
    avg_Ein_wlin = np.mean(Eins_wlin)
    avg_Eout_wlin = np.mean(Eouts_wlin)
    avg_acc = np.mean(accs)
    Ein_avg = np.mean(Eins_wt, axis=0)
    Eout_avg = np.mean(Eouts_wt, axis=0)
    avg_Ein_acc = np.mean(Ein_accs)
    avg_Eout_acc = np.mean(Eout_accs)

    # Plot errors
    plot_errors(Ein_avg, Eout_avg, avg_Ein_wlin, avg_Eout_wlin, interval, iterations)
    print('average accuracy:', avg_acc)
    print('average Ein accuracy:', avg_Ein_acc)
    print('average Eout accuracy:', avg_Eout_acc)

    return Eins_wlin, Eouts_wlin

def plot11(Ein_wlin, Ein_wpoly):
    Ein_squared_diff = []
    for i in range(len(Ein_wlin)):
        Ein_squared_diff.append(Ein_wlin[i] - Ein_wpoly[i])
    Ein_squared_diff_avg = np.mean(Ein_squared_diff)
    print(Ein_squared_diff_avg)
    plt.hist(Ein_squared_diff, bins=30, alpha=0.5, label="Ein_wlin^2 - Ein_wpoly^2")
    plt.xlabel("Ein_wlin^2 - Ein_wpoly^2")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Frequency of Ein_wlin^2 - Ein_wpoly^2 over Iterations")
    plt.show()

def plot12(Eout_wlin, Eout_wpoly):
    Eout_squared_diff = []
    for i in range(len(Eout_wlin)):
      Eout_squared_diff.append(Eout_wlin[i] - Eout_wpoly[i])
    Eout_squared_diff_avg = np.mean(Eout_squared_diff)
    print(Eout_squared_diff_avg)
    plt.hist(Eout_squared_diff, bins=30, alpha=0.5, label="Eout_wlin^2 - Eout_wpoly^2")
    plt.xlabel("Eout_wlin^2 - Eout_wpoly^2")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Frequency of Eout_wlin^2 - Eout_wpoly^2 over Iterations")
    plt.show()
        

def q12():
    Ein_wlin, Eout_wlin = q10()
    Ein_wpoly, Eout_wpoly = q11()

    plot11(Ein_wlin, Ein_wpoly)
    plot12(Eout_wlin, Eout_wpoly)
        
q10()


