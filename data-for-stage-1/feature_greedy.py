import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset
file_path = 'train_data.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Prepare the dataset
X = data.drop(columns=['id', 'home_team_win'])
y = data['home_team_win'].astype(int)

# Encode categorical features
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    X[col] = label_encoders[col].fit_transform(X[col].astype(str))

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Greedy feature selection
selected_features = []
remaining_features = list(X_train.columns)
best_accuracy = 0

while remaining_features:
    feature_accuracies = {}
    for feature in remaining_features:
        # Train a model with the current set of selected features + this feature
        temp_features = selected_features + [feature]
        model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, min_samples_leaf=2,
                                       min_samples_split=2)
        model.fit(X_train[temp_features], y_train)
        y_pred = model.predict(X_test[temp_features])
        feature_accuracies[feature] = accuracy_score(y_test, y_pred)

    # Find the best feature to add
    best_feature = max(feature_accuracies, key=feature_accuracies.get)
    best_feature_accuracy = feature_accuracies[best_feature]

    if best_feature_accuracy > best_accuracy:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_accuracy = best_feature_accuracy
        print(f"Added feature: {best_feature}, New Accuracy: {best_feature_accuracy:.4f}")
    else:
        # Stop if no improvement
        break

print(f"Selected features: {selected_features}")
print(f"Final accuracy: {best_accuracy:.4f}")

"""
Output:

/Users/chuajerome/anaconda3/bin/python /Users/chuajerome/Desktop/ML_Project/data-for-stage-1/feature_greedy.py 
Added feature: away_team_season, New Accuracy: 0.5601
Added feature: home_pitching_H_batters_faced_10RA, New Accuracy: 0.5623
Added feature: home_team_wins_std, New Accuracy: 0.5655
Added feature: away_team_abbr, New Accuracy: 0.5732
Added feature: home_pitcher_H_batters_faced_mean, New Accuracy: 0.5754
Added feature: is_night_game, New Accuracy: 0.5781
Added feature: away_team_spread_mean, New Accuracy: 0.5786
Added feature: home_batting_batting_avg_skew, New Accuracy: 0.5863
Selected features: ['away_team_season', 'home_pitching_H_batters_faced_10RA', 'home_team_wins_std', 'away_team_abbr', 'home_pitcher_H_batters_faced_mean', 'is_night_game', 'away_team_spread_mean', 'home_batting_batting_avg_skew']
Final accuracy: 0.5863

Process finished with exit code 0


"""