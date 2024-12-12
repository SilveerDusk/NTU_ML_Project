# Import Necessary Libraries
import pandas as pd
import numpy as np

# For data preprocessing and modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# For saving the model
import joblib

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Data
data = pd.read_csv('train_data.csv')

# Display the first few rows
print("Initial Data Snapshot:")
print(data.head())

# 2. Data Preprocessing

# a. Drop Unnecessary Categorical Columns
# Based on your initial code, you intended to drop the following columns
categorical_columns_to_drop = ['id', 'home_team_abbr', 'away_team_abbr', 'date',
                               'testing', 'home_pitcher', 'away_pitcher',
                               'home_team_season', 'away_team_season']

# However, it's often beneficial to encode categorical variables instead of dropping them.
# Here, we'll proceed by encoding them.

# b. Feature Engineering from 'date' Column
# Convert 'date' to datetime
# data['date'] = pd.to_datetime(data['date'])
#
# # Extract meaningful features
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day_of_week'] = data['date'].dt.dayofweek
# data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Drop the original 'date' column
data.drop(columns=['date', 'id'], inplace=True)

# c. Encode Categorical Variables
# Identify categorical columns to encode
categorical_columns = ['home_team_abbr', 'away_team_abbr', 'home_pitcher',
                       'away_pitcher', 'home_team_season', 'away_team_season']

# Initialize OneHotEncoder for nominal categorical variables
# For high-cardinality features like 'home_pitcher' and 'away_pitcher', One-Hot Encoding might lead to high dimensionality.
# Consider using Target Encoding or frequency encoding if necessary. For simplicity, we'll proceed with One-Hot Encoding.

# d. Handle 'is_night_game' Feature
# Convert boolean to integer
data['is_night_game'] = data['is_night_game'].map({True: 1, False: 0})

# e. Define Numerical Columns
# Exclude target and categorical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# f. Split Features and Target
X = data.drop('home_team_win', axis=1)
y = data['home_team_win'].astype(int)  # Convert boolean to integer (1 for True, 0 for False)

# 3. Define Preprocessing Steps

# Preprocessing for numerical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
    ('scaler', StandardScaler())  # Scale features
])

from sklearn.preprocessing import OneHotEncoder

# Update the pipeline step for OneHotEncoder:
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-Hot Encode
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ]
)

# 4. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Ensures proportional representation of classes
)

print(f"\nTraining Set: {X_train.shape}")
print(f"Testing Set: {X_test.shape}")

# 5. Create the Modeling Pipeline
# We'll use a Random Forest Classifier. You can experiment with other models as needed.

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Train the Model
model_pipeline.fit(X_train, y_train)
print("\nModel training completed.")

# 7. Make Predictions
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# 8. Evaluate the Model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 9. Feature Importance Analysis
# To understand which features are most important in the model's decisions.

# Extract the classifier from the pipeline
classifier = model_pipeline.named_steps['classifier']

# Extract feature names after One-Hot Encoding
onehot_feature_names = model_pipeline.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .named_steps['onehot'] \
    .get_feature_names_out(categorical_columns)

# Combine numerical and categorical feature names
all_feature_names = numerical_columns + list(onehot_feature_names)

# Create a Series for feature importances
feature_importances = pd.Series(classifier.feature_importances_, index=all_feature_names)

# Sort feature importances in descending order
feature_importances = feature_importances.sort_values(ascending=False)

# Plot top 20 feature importances
plt.figure(figsize=(10,8))
sns.barplot(x=feature_importances[:20], y=feature_importances[:20].index)
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 10. Hyperparameter Tuning (Optional but Recommended)
# Improve model performance by tuning hyperparameters using GridSearchCV.

# Define parameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

# Perform Grid Search
print("\nStarting Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)
print("Grid Search completed.")

# Best parameters from Grid Search
print(f"\nBest Parameters: {grid_search.best_params_}")

# Best ROC-AUC Score from Grid Search
print(f"Best ROC-AUC Score from Grid Search: {grid_search.best_score_:.4f}")

# 11. Evaluate the Best Model from Grid Search
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))

roc_auc_best = roc_auc_score(y_test, y_proba_best)
print(f"ROC-AUC Score for Best Model: {roc_auc_best:.4f}")

# Plot ROC Curve for Best Model
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_proba_best)
plt.figure(figsize=(8,6))
plt.plot(fpr_best, tpr_best, label=f'Best Random Forest (AUC = {roc_auc_best:.4f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 12. Save the Trained Model
# This allows you to load the model later without retraining.

model_filename = 'home_team_win_predictor.pkl'
joblib.dump(best_model, model_filename)
print(f"\nTrained model saved as {model_filename}")

# 13. Load the Saved Model (Optional)
# To verify that the model can be loaded and used for predictions.

# loaded_model = joblib.load(model_filename)
# sample_prediction = loaded_model.predict(X_test.iloc[0:1])
# print(f"\nSample Prediction for the first test instance: {sample_prediction[0]}")

# Load the test data
test_data = pd.read_csv('same_season_test_data.csv')


# Map 'is_night_game' to integer
test_data['is_night_game'] = test_data['is_night_game'].map({True: 1, False: 0})

# Ensure all expected columns are present
# Extract the expected columns from the preprocessing pipeline
categorical_columns = ['home_team_abbr', 'away_team_abbr', 'home_pitcher',
                       'away_pitcher', 'home_team_season', 'away_team_season']
numerical_columns = test_data.select_dtypes(include=[np.number]).columns.tolist()

# Drop any columns not in the model's preprocessing pipeline
expected_columns = categorical_columns + numerical_columns
test_data = test_data.reindex(columns=expected_columns, fill_value=0)

# Extract the 'id' column for saving predictions
ids = test_data['id']  # Save for later
X_test_final = test_data.drop(columns=['id'], errors='ignore')

# Make predictions
test_predictions = best_model.predict(X_test_final)

# Save predictions to a CSV file
output_df = pd.DataFrame({'id': ids, 'home_team_win': test_predictions})
output_filename = 'predictions.csv'
output_df.to_csv(output_filename, index=False)

print(f"Predictions saved to {output_filename}")