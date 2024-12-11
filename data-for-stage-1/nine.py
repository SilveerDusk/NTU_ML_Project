import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('train_data.csv')

# Convert 'date' to datetime and create a numeric representation (if date is important)
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data['date_numeric'] = (data['date'] - data['date'].min()).dt.days
    data = data.drop(columns=['date'])  # Drop the original date column

# Convert categorical columns to numeric using Label Encoding
encoder = LabelEncoder()
data['home_team_abbr'] = encoder.fit_transform(data['home_team_abbr'])
data['away_team_abbr'] = encoder.fit_transform(data['away_team_abbr'])

# Define features and target variable
X = data.drop(columns=['home_team_win'])  # Features
y = data['home_team_win']  # Target

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')

# ANN model
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'MLP Accuracy: {accuracy_mlp:.2f}')

# Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')

# Model evaluation
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\nANN Classification Report:\n", classification_report(y_test, y_pred_mlp))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
