import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_data = pd.read_csv("train_data.csv")

# Define selected features
selected_features = [
    'away_team_season',
    'home_pitching_H_batters_faced_10RA',
    'home_team_wins_std',
    'away_team_abbr',
    'home_pitcher_H_batters_faced_mean',
    'is_night_game',
    'away_team_spread_mean',
    'home_batting_batting_avg_skew'
]

# Prepare data
X = train_data[selected_features]
y = train_data['home_team_win']  # Target variable

# Encode categorical columns
print("Encoding categorical columns...")
for column in X.select_dtypes(include=['object']).columns:
    print(f"Encoding column: {column}")
    X[column] = LabelEncoder().fit_transform(X[column])

# Handle missing values
print("Handling missing values...")
X = X.fillna(X.mean())

# Normalize features
print("Normalizing features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Hyperparameter tuning with GridSearchCV
print("Starting hyperparameter tuning...")
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_svm = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Make predictions
y_pred = best_svm.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)

print("Final Model Accuracy:", accuracy)
print("Classification Report:\n", classification)
