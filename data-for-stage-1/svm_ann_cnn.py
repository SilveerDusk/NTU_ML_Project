# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Handling Imbalanced Data
from imblearn.over_sampling import SMOTE

# Models
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.optimizers import Adam

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensemble
from sklearn.ensemble import VotingClassifier

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. Data Loading and Initial Preprocessing
# ------------------------------------------

# Load the dataset
data = pd.read_csv('train_data.csv')

# Convert 'is_night_game' to 1 for True, 0 for False
if 'is_night_game' in data.columns:
    data['is_night_game'] = data['is_night_game'].fillna(False).astype(int)

# Drop categorical columns
categorical_columns = ['id', 'home_team_abbr', 'away_team_abbr', 'date', 'testing',
                       "home_pitcher", "away_pitcher", "home_team_season", "away_team_season"]
data = data.drop(columns=categorical_columns, errors='ignore')

# 2. Handling Missing Values with KNN Imputation
# -----------------------------------------------

# Separate features and target
X = data.drop(columns=['home_team_win'], errors='ignore')
y = data['home_team_win'].astype(int)

# Initialize KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)

# Fit and transform the data
X_imputed = knn_imputer.fit_transform(X)

# Convert back to DataFrame for easier handling
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# 3. Handling Class Imbalance with SMOTE
# ---------------------------------------

# Check class distribution
print("Original class distribution:")
print(y.value_counts())

# Initialize SMOTE
smote = SMOTE(random_state=RANDOM_STATE)

# Apply SMOTE to the imputed data
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# Check new class distribution
print("\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts())

# 4. Feature Scaling with StandardScaler
# --------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# 5. Feature Selection using Random Forest Feature Importance
# ------------------------------------------------------------

# Initialize Random Forest for feature selection
rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

# Fit the model
rf_selector.fit(X_scaled, y_resampled)

# Select features with importance greater than the mean importance
selector = SelectFromModel(rf_selector, prefit=True, threshold='mean')
X_selected = selector.transform(X_scaled)

# Get selected feature names
selected_features = X_imputed.columns[selector.get_support()]
print(f"\nSelected Features ({len(selected_features)}): {list(selected_features)}")

# 6. Train-Test Split with Stratification
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_resampled, test_size=0.2, random_state=RANDOM_STATE, stratify=y_resampled
)

# 7. Hyperparameter Tuning for SVM using GridSearchCV
# ---------------------------------------------------

# Define parameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Initialize SVM
svm = SVC(probability=True, random_state=RANDOM_STATE)

# Initialize GridSearchCV
svm_grid = GridSearchCV(estimator=svm, param_grid=svm_param_grid,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                        scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
svm_grid.fit(X_train, y_train)

# Best parameters and estimator
print("\nBest SVM Parameters:", svm_grid.best_params_)
best_svm = svm_grid.best_estimator_


# 8. Building Improved ANN Model
# -------------------------------

def build_ann(input_dim, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Wrap the ANN model for use in scikit-learn
ann_model = KerasClassifier(build_fn=build_ann, input_dim=X_train.shape[1],
                            verbose=0)

# Define parameter grid for ANN
ann_param_grid = {
    'epochs': [50, 100],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'dropout_rate': [0.3, 0.5]
}

# Initialize GridSearchCV for ANN
ann_grid = GridSearchCV(estimator=ann_model, param_grid=ann_param_grid,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                        scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV for ANN
ann_grid.fit(X_train, y_train)

# Best parameters and estimator for ANN
print("\nBest ANN Parameters:", ann_grid.best_params_)
best_ann = ann_grid.best_estimator_


# 9. Building Improved 1D CNN Model
# ----------------------------------

def build_cnn(input_shape, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Reshape data for CNN
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# Wrap the CNN model for use in scikit-learn
cnn_model = KerasClassifier(build_fn=build_cnn, input_shape=(X_train.shape[1], 1),
                            verbose=0)

# Define parameter grid for CNN
cnn_param_grid = {
    'epochs': [50, 100],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'dropout_rate': [0.3, 0.5]
}

# Initialize GridSearchCV for CNN
cnn_grid = GridSearchCV(estimator=cnn_model, param_grid=cnn_param_grid,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                        scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV for CNN
cnn_grid.fit(X_train_cnn, y_train)

# Best parameters and estimator for CNN
print("\nBest CNN Parameters:", cnn_grid.best_params_)
best_cnn = cnn_grid.best_estimator_

# 10. Training the Best ANN Model with Early Stopping
# ----------------------------------------------------

# Rebuild the best ANN model with optimal hyperparameters
best_ann_params = ann_grid.best_params_
ann_final = build_ann(input_dim=X_train.shape[1],
                      learning_rate=best_ann_params['learning_rate'],
                      dropout_rate=best_ann_params['dropout_rate'])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_ann_model.h5', monitor='val_loss', save_best_only=True)

# Train the ANN model
history_ann = ann_final.fit(X_train, y_train,
                            epochs=best_ann_params['epochs'],
                            batch_size=best_ann_params['batch_size'],
                            validation_split=0.2,
                            callbacks=[early_stop, checkpoint],
                            verbose=1)

# 11. Training the Best CNN Model with Early Stopping
# ----------------------------------------------------

# Rebuild the best CNN model with optimal hyperparameters
best_cnn_params = cnn_grid.best_params_
cnn_final = build_cnn(input_shape=(X_train.shape[1], 1),
                      learning_rate=best_cnn_params['learning_rate'],
                      dropout_rate=best_cnn_params['dropout_rate'])

# Define callbacks
early_stop_cnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_cnn = ModelCheckpoint('best_cnn_model.h5', monitor='val_loss', save_best_only=True)

# Train the CNN model
history_cnn = cnn_final.fit(X_train_cnn, y_train,
                            epochs=best_cnn_params['epochs'],
                            batch_size=best_cnn_params['batch_size'],
                            validation_split=0.2,
                            callbacks=[early_stop_cnn, checkpoint_cnn],
                            verbose=1)

# 12. Evaluating the Models
# --------------------------

# Evaluate SVM
svm_preds = best_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"\nSVM Accuracy: {svm_accuracy:.4f}")
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))

# Evaluate ANN
ann_preds = (ann_final.predict(X_test) > 0.5).astype(int).flatten()
ann_accuracy = accuracy_score(y_test, ann_preds)
print(f"\nANN Accuracy: {ann_accuracy:.4f}")
print("ANN Classification Report:")
print(classification_report(y_test, ann_preds))

# Evaluate CNN
cnn_preds = (cnn_final.predict(X_test_cnn) > 0.5).astype(int).flatten()
cnn_accuracy = accuracy_score(y_test, cnn_preds)
print(f"\n1DCNN Accuracy: {cnn_accuracy:.4f}")
print("1DCNN Classification Report:")
print(classification_report(y_test, cnn_preds))


# 13. Ensemble Method using Voting Classifier
# -------------------------------------------

# Define a Voting Classifier with SVM, ANN, and CNN
# Note: For CNN and ANN, we need to define scikit-learn compatible estimators

# Create wrapper functions for ANN and CNN predictions
class KerasANNWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)


class KerasCNNWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)


# Initialize wrappers
ann_wrapper = KerasANNWrapper(ann_final)
cnn_wrapper = KerasCNNWrapper(cnn_final)

# Since VotingClassifier requires estimators to have fit and predict methods,
# we need to create dummy estimators. Alternatively, use Soft Voting with probabilities.

# Create custom estimators
from sklearn.base import BaseEstimator, ClassifierMixin


class ANNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)


class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)


# Initialize custom classifiers
ann_classifier = ANNClassifier(ann_final)
cnn_classifier = CNNClassifier(cnn_final)

# Initialize Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('svm', best_svm),
        ('ann', ann_classifier),
        ('cnn', cnn_classifier)
    ],
    voting='soft'  # Use 'soft' voting to average predicted probabilities
)


# Since ANN and CNN are already trained, we need to fit a dummy estimator
# Here, we override the fit method to do nothing
def dummy_fit(self, X, y):
    return self


# Assign dummy fit
VotingClassifier.fit = dummy_fit

# Fit the Voting Classifier
voting_clf.fit(X_train, y_train)

# Predict with Voting Classifier
ensemble_preds = voting_clf.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
print(f"\nEnsemble (Voting) Accuracy: {ensemble_accuracy:.4f}")
print("Ensemble Classification Report:")
print(classification_report(y_test, ensemble_preds))

# 14. Final Results
# -----------------

print(
    f"\nFinal Accuracies:\nSVM: {svm_accuracy:.4f}\nANN: {ann_accuracy:.4f}\n1DCNN: {cnn_accuracy:.4f}\nEnsemble: {ensemble_accuracy:.4f}")

# 15. Plotting Training History (Optional)
# ----------------------------------------

# Plot ANN training history
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history_ann.history['accuracy'], label='Train Accuracy')
plt.plot(history_ann.history['val_accuracy'], label='Validation Accuracy')
plt.title('ANN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot CNN training history
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 16. Confusion Matrix (Optional)
# --------------------------------

# Function to plot confusion matrix
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()


# Plot confusion matrices
plot_confusion(y_test, svm_preds, "SVM Confusion Matrix")
plot_confusion(y_test, ann_preds, "ANN Confusion Matrix")
plot_confusion(y_test, cnn_preds, "1DCNN Confusion Matrix")
plot_confusion(y_test, ensemble_preds, "Ensemble Confusion Matrix")
