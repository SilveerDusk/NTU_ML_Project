import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
import numpy as np
from scipy.spatial import distance

# Load the dataset
data = pd.read_csv('train_data.csv')

# Convert 'is_night_game' to 1 for True, 0 for False
if 'is_night_game' in data.columns:
    data['is_night_game'] = data['is_night_game'].fillna(False).astype(int)

# Drop categorical columns
categorical_columns = ['id', 'home_team_abbr', 'away_team_abbr', 'date', 'testing', "home_pitcher", "away_pitcher", "home_team_season", "away_team_season"]
data = data.drop(columns=categorical_columns, errors='ignore')

# Handle missing values
data = data.fillna(0)



# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.drop(columns=['home_team_win']))
labels = data['home_team_win'].values

# Ensure labels are binary
labels = labels.astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)

# Feature selection using ReliefF implementation
def reliefF(X, y, k=5):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for i in range(n_samples):
        distances = distance.cdist([X[i]], X, metric='euclidean').flatten()
        nearest_hits = np.argsort(distances[y == y[i]])[1:k+1]
        nearest_misses = np.argsort(distances[y != y[i]])[:k]

        for hit in nearest_hits:
            weights += np.abs(X[i] - X[hit])

        for miss in nearest_misses:
            weights -= np.abs(X[i] - X[miss])

    return np.argsort(weights)[::-1][:k]  # Select top-k features

selected_features = reliefF(X_train, y_train)
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# 1DCNN Model
cnn_model = Sequential([
    Input(shape=(X_train_selected.shape[1], 1)),
    Conv1D(filters=16, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape for 1D CNN
X_train_cnn = X_train_selected[..., np.newaxis]
X_test_cnn = X_test_selected[..., np.newaxis]

cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
cnn_preds = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
cnn_accuracy = accuracy_score(y_test, cnn_preds)
print("1DCNN Accuracy:", cnn_accuracy)

# ANN Model
ann_model = Sequential([
    Input(shape=(X_train_selected.shape[1],)),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann_model.fit(X_train_selected, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
ann_preds = (ann_model.predict(X_test_selected) > 0.5).astype(int)
ann_accuracy = accuracy_score(y_test, ann_preds)
print("ANN Accuracy:", ann_accuracy)

# SVM Model
svm_model = SVC(kernel='rbf', C=1000, gamma=0.1, probability=True)
svm_model.fit(X_train_selected, y_train)
svm_preds = svm_model.predict(X_test_selected)
svm_accuracy = accuracy_score(y_test, svm_preds)
print("SVM Accuracy:", svm_accuracy)

# Final Results
print(f"Final Accuracies: 1DCNN={cnn_accuracy:.4f}, ANN={ann_accuracy:.4f}, SVM={svm_accuracy:.4f}")