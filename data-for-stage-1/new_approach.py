import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the dataset
print("Loading data...")
data = pd.read_csv("train_data.csv")  # Replace with your file name

# Calculate rolling Win% for home and away teams
print("Calculating rolling Win%...")
data['home_team_win_count'] = data.groupby('home_team_abbr', group_keys=False)['home_team_win'].cumsum()
data['home_team_game_count'] = data.groupby('home_team_abbr', group_keys=False).cumcount() + 1
data['home_team_win_percentage'] = data['home_team_win_count'] / data['home_team_game_count']

data['away_team_win_count'] = data.groupby('away_team_abbr', group_keys=False)['home_team_win'].apply(lambda x: (~x).cumsum())
data['away_team_game_count'] = data.groupby('away_team_abbr', group_keys=False).cumcount() + 1
data['away_team_win_percentage'] = data['away_team_win_count'] / data['away_team_game_count']

# Combined Win% feature
data['win_percentage'] = data['home_team_win_percentage'] / (
    data['home_team_win_percentage'] + data['away_team_win_percentage'] + 1e-9
)

# Rolling averages for other metrics
print("Calculating rolling averages for other features...")
metrics = [
    'home_batting_batting_avg_10RA', 'away_batting_batting_avg_10RA',
    'home_batting_RBI_10RA', 'away_batting_RBI_10RA',
    'home_pitching_earned_run_avg_10RA', 'away_pitching_earned_run_avg_10RA', 'is_night_game'
]

for metric in metrics:
    data[f'{metric}_rolling'] = (
        data.groupby('home_team_abbr' if 'home' in metric else 'away_team_abbr', group_keys=False)[metric]
        .apply(lambda x: x.shift(1).expanding().mean())
    )

# Fill missing values instead of dropping rows
print("Filling missing values...")
data.fillna(data.median(numeric_only=True), inplace=True)

# Prepare data for modeling
print("Preparing data for training and testing...")
target = 'home_team_win'  # Target column
features = ['win_percentage'] + [f'{metric}_rolling' for metric in metrics]

X = data[features]
y = data[target]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Ensure sufficient samples
print(f"Total samples: {len(data)}, Target distribution:\n{y.value_counts()}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Random Forest model
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)
