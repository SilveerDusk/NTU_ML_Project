import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

# Refer to paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8871522/pdf/entropy-24-00288.pdf

# Load datasets
print("Loading training and testing datasets...")
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("same_season_test_data.csv")
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Function to prepare data
def prepare_data(data):
    print("Preparing data by dropping unnecessary columns and separating features and target...")
    X = data.drop(columns=['id', 'home_team_win'], errors='ignore')
    y = data['home_team_win'] if 'home_team_win' in data.columns else None
    print(f"Features shape: {X.shape}, Target shape: {y.shape if y is not None else 'N/A'}")
    return X, y

# Function to encode categorical columns
def encode_categorical_columns(df):
    print("Encoding categorical columns using LabelEncoder...")
    df = df.copy()
    for column in df.select_dtypes(include=['object']).columns:
        print(f"Encoding column: {column}")
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

# Function to handle missing values
def handle_missing_values(df):
    print("Handling missing values by filling with column mean...")
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values before imputation: {missing_count}")
    df_filled = df.fillna(df.mean())
    missing_count_after = df_filled.isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_count_after}")
    return df_filled

# Function to normalize the data
def normalize_data(X_train, X_test):
    print("Normalizing features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Feature scaling completed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled

# Prepare and encode the data
X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)
X_train_encoded = encode_categorical_columns(X_train)
X_test_encoded = encode_categorical_columns(X_test)

# Align features by ensuring both datasets have the same columns
print("Aligning features between training and testing datasets...")
common_columns = list(set(X_train_encoded.columns).intersection(X_test_encoded.columns))
print(f"Number of common columns: {len(common_columns)}")
X_train_aligned = X_train_encoded[common_columns]
X_test_aligned = X_test_encoded[common_columns]

# Handle missing values
X_train_imputed = handle_missing_values(X_train_aligned)
X_test_imputed = handle_missing_values(X_test_aligned)

# Normalize features
X_train_scaled, X_test_scaled = normalize_data(X_train_imputed, X_test_imputed)

# Initialize Recursive Feature Elimination (RFE)
print("Performing Recursive Feature Elimination (RFE) to select top features...")
svc = SVC(kernel='linear', random_state=42)
rfe = RFE(estimator=svc, n_features_to_select=20)  # Select top 20 features
X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
X_test_selected = rfe.transform(X_test_scaled)
print(f"Selected features shape (Train): {X_train_selected.shape}, (Test): {X_test_selected.shape}")

# Hyperparameter tuning using GridSearchCV
print("Starting hyperparameter tuning with GridSearchCV...")
param_grid = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 10, 100]
}
grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)
print("GridSearchCV completed. Best parameters found:")
print(grid_search.best_params_)

# Get the best model
best_svm = grid_search.best_estimator_

# Make predictions
print("Making predictions on the test dataset...")
y_pred = best_svm.predict(X_test_selected)
y_pred_prob = best_svm.predict_proba(X_test_selected)[:, 1] if hasattr(best_svm, 'predict_proba') else None

# Evaluate results
print("Evaluating model performance...")
results = {
    'accuracy': accuracy_score(y_test, y_pred) if y_test is not None else 'N/A',
    'roc_auc': roc_auc_score(y_test, y_pred_prob) if y_test is not None and y_pred_prob is not None else 'N/A',
    'classification_report': classification_report(y_test, y_pred) if y_test is not None else 'N/A',
    'best_params': grid_search.best_params_
}

# Output results
print("Results:")
print(results)
