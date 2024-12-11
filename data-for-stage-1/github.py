import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Load the processed dataset
dataset_path = 'train_data.csv'
data = pd.read_csv(dataset_path)

# Extract date components
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday

# Convert 'home_team_win' from boolean to numeric (1 for True, 0 for False)
data['home_team_win'] = data['home_team_win'].astype(int)

# Check for missing values and handle them
data.fillna(data.median(numeric_only=True), inplace=True)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Define features and target
X = data.drop(columns=['home_team_win', 'date'])  # Exclude 'date' column
y = data['home_team_win']

# Clip extreme values to reduce outliers for numeric columns only
numeric_columns = X.select_dtypes(include=['number']).columns
X[numeric_columns] = X[numeric_columns].clip(lower=X[numeric_columns].quantile(0.01),
                                             upper=X[numeric_columns].quantile(0.99),
                                             axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('home_team_win_model.h5')

# Save the scaler for future use
import joblib
joblib.dump(scaler, 'scaler.pkl')

# Output the paths to the model and scaler for reference
print("Model saved as: home_team_win_model.h5")
print("Scaler saved as: scaler.pkl")
