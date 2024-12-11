import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'train_data.csv' with your actual dataset file
data = pd.read_csv('train_data.csv')

# Drop unnecessary columns, e.g., 'id' and others not needed
data = data.drop(['id'], axis=1)

# Handle missing values and encode 'is_night_game'
# Encode 'is_night_game': True -> 1, False -> 0, Missing -> -1
data['is_night_game'] = data['is_night_game'].map({True: 1, False: 0}).fillna(-1)

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=['object', 'category']).columns
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

# Impute missing values for numerical features
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_features] = numerical_imputer.fit_transform(data[numerical_features])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Combine encoded categorical features and numerical features
X = pd.DataFrame(
    np.hstack([data[numerical_features].values, encoded_categorical]),
    columns=list(numerical_features) + list(encoded_feature_names)
)

# Separate target variable (y)
y = data['home_team_win']  # Binary target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top features
top_features = feature_importances.head(100)  # Adjust number of features as needed
print("Top Features:")
print(top_features)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(top_features['Feature'], top_features['Importance'], align='center')
plt.xlabel("Importance")
plt.title("Top 20 Feature Importances for Predicting 'home_team_win'")
plt.gca().invert_yaxis()
plt.show()

# Save feature importance results to CSV
feature_importances.to_csv('feature_importances.csv', index=False)
print("Feature importances saved to 'feature_importances.csv'")
