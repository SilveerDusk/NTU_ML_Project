import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
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

# Encode categorical variables
categorical_columns = ['away_team_season', 'away_team_abbr']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cols = encoder.fit_transform(X[categorical_columns])

# Add encoded columns back to the DataFrame and drop originals
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_columns))
X = X.drop(columns=categorical_columns).reset_index(drop=True)
X = pd.concat([X, encoded_df], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize XGBoost model
model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_res, y_train_res)
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

# Train the best model with early stopping
best_model.fit(
    X_train_res,
    y_train_res,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Feature Importance Visualization
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importances')
plt.show()

# Detailed classification report
print(classification_report(y_test, y_pred))
