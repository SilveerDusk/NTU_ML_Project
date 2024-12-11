import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import optuna

# Step 1: Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv("train_data.csv")

# Drop the 'id' column if it exists
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Separate features and target
X = data.drop(columns=['home_team_win'], errors='ignore')  # Drop the target variable
y = data['home_team_win']  # Target variable

# Handle missing values with median imputation
print("Handling missing values...")
X = X.fillna(X.median(numeric_only=True))

# Encode categorical features
print("Encoding categorical features...")
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical features

# Add interaction features (example)
# X['interaction_feature'] = X['feature1'] * X['feature2'] # Uncomment and replace 'feature1'/'feature2'

# Scale the features
print("Scaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Balance the dataset using SMOTE
print("Balancing dataset with SMOTE...")
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# Step 2: Feature selection using LightGBM
print("Selecting important features with LightGBM...")
lgbm = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
lgbm.fit(X_train_bal, y_train_bal)

# Get top features
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': lgbm.feature_importances_
}).sort_values(by='importance', ascending=False)

top_k = 30
top_features = feature_importances.head(top_k)['feature'].tolist()
print(f"Top {top_k} features selected: {top_features}")

X_train_top = X_train_bal[:, :top_k]
X_test_top = X_test[:, :top_k]

# Step 3: Hyperparameter tuning with Optuna for LightGBM
print("Tuning LightGBM with Optuna...")
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
    }
    model = LGBMClassifier(**param, random_state=42)
    scores = cross_val_score(model, X_train_top, y_train_bal, cv=5, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best LightGBM Parameters: {best_params}")

lgbm_optimized = LGBMClassifier(**best_params, random_state=42)
lgbm_optimized.fit(X_train_top, y_train_bal)

# Step 4: Train SVM with hyperparameter tuning
print("Tuning SVM...")
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}
svm = SVC(probability=True, random_state=42)
svm_grid = RandomizedSearchCV(svm, param_distributions=svm_param_grid, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_top, y_train_bal)

best_svm = svm_grid.best_estimator_
print("Best SVM Parameters:", svm_grid.best_params_)

# Step 5: Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_top, y_train_bal)

# Step 6: Ensemble with Voting Classifier
print("Ensembling models...")
voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_optimized),
        ('rf', rf),
        ('svm', best_svm)
    ],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_top, y_train_bal)

# Step 7: Evaluate the ensemble model
print("Evaluating ensemble model...")
y_pred = voting_clf.predict(X_test_top)
y_pred_prob = voting_clf.predict_proba(X_test_top)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)
