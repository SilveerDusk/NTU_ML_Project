import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('train_data.csv')

# Selected features
features = [
    'away_team_season', 'home_pitching_H_batters_faced_10RA', 'home_team_wins_std',
    'away_team_abbr', 'home_pitcher_H_batters_faced_mean', 'is_night_game',
    'away_team_spread_mean', 'home_batting_batting_avg_skew'
]
X = data[features].copy()
y = data['home_team_win']

# Fix missing values and ensure correct data types
data['is_night_game'] = data['is_night_game'].fillna(0).astype(int)
data['away_team_season'] = data['away_team_season'].astype('category').cat.codes
data['away_team_abbr'] = data['away_team_abbr'].astype('category').cat.codes

# Update X with corrected columns
X['is_night_game'] = data['is_night_game']
X['away_team_season'] = data['away_team_season']
X['away_team_abbr'] = data['away_team_abbr']

# Handle missing values using imputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Add polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(X.columns))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)

# Perform hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, verbose=1)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Train the best model
best_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
