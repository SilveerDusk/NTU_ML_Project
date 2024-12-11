import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from skrebate import ReliefF

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                     Flatten, BatchNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.impute import SimpleImputer  # Added for imputation

import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# 1. Data Collection and Preprocessing
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Display the first few rows
    print("Dataset Preview:")
    print(data.head())

    # Derived features
    data['home_batting_efficiency'] = data['home_batting_onbase_plus_slugging_mean'] / (
                data['home_batting_batting_avg_mean'] + 1e-5)
    data['away_batting_efficiency'] = data['away_batting_onbase_plus_slugging_mean'] / (
                data['away_batting_batting_avg_mean'] + 1e-5)
    data['home_wins_variability'] = data['home_team_wins_std'] / (data['home_team_wins_mean'] + 1e-5)
    data['away_wins_variability'] = data['away_team_wins_std'] / (data['away_team_wins_mean'] + 1e-5)
    data['home_normalized_batting_avg'] = (data['home_batting_batting_avg_mean'] - data[
        'home_batting_batting_avg_skew']) / (data['home_batting_batting_avg_std'] + 1e-5)
    data['away_normalized_batting_avg'] = (data['away_batting_batting_avg_mean'] - data[
        'away_batting_batting_avg_skew']) / (data['away_batting_batting_avg_std'] + 1e-5)
    data['batting_avg_interaction'] = data['home_batting_batting_avg_mean'] * data['away_batting_batting_avg_mean']
    data['pitching_avg_interaction'] = data['home_pitching_H_batters_faced_mean'] * data[
        'away_pitching_H_batters_faced_mean']
    data['game_context'] = data['is_night_game'] * (data['home_team_abbr'] == data['away_team_abbr']).astype(int)

    # Identify all feature columns (excluding 'id', 'home_team_abbr', 'away_team_abbr', 'date', 'testing', 'home_team_win')
    feature_columns = data.columns.drop(['id', 'home_team_abbr', 'away_team_abbr', 'date', 'home_team_win'])

    # Separate numeric and categorical columns
    numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = list(set(feature_columns) - set(numeric_features))

    # Initialize the imputer (mean strategy for numerical features)
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the numeric feature columns
    data[numeric_features] = imputer.fit_transform(data[numeric_features])

    # Handle categorical features (if any)
    # Option 1: Drop categorical features
    # data = data.drop(columns=categorical_features)

    # Option 2: Encode categorical features (e.g., Label Encoding)
    if categorical_features:
        print(f"Found categorical features: {categorical_features}")
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        for col in categorical_features:
            data[col] = label_encoder.fit_transform(data[col].astype(str))
        print("Categorical features have been label encoded.")

    # Alternatively, for more complex encoding, consider One-Hot Encoding:
    # data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    return data


# 2. Feature Selection using ReliefF
def select_features_relieff(X, y, n_features):
    # Initialize ReliefF
    relief = ReliefF(n_features_to_select=n_features, n_neighbors=100, discrete_threshold=10)
    relief.fit(X, y)
    # Get the selected feature indices
    selected_indices = relief.top_features_[:n_features]
    return selected_indices


# 3. Model Definitions
def build_ann_model(input_dim, optimizer='adam'):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    else:
        opt = RMSprop(learning_rate=0.001)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_1dcnn_model(input_shape, optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    else:
        opt = RMSprop(learning_rate=0.001)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 4. Evaluation Function
def evaluate_model(y_test, y_pred, y_pred_proba):
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    return acc, conf_matrix, class_report, roc_auc


# 5. Main Function
def main():
    # File path to your dataset
    file_path = 'train_data.csv'

    # Load and preprocess data
    print("running")
    data = load_and_preprocess_data(file_path)
    print("reading in")

    # Define label
    label = 'home_team_win'

    # Encode the target variable if it's categorical
    label_encoder = LabelEncoder()
    data[label] = label_encoder.fit_transform(data[label])

    # Define feature sets for Dataset 1 and Dataset 2
    # Dataset 1: Using only starting pitcher (SP) data
    # Assuming 'home_pitcher_*' and 'away_pitcher_*' columns are SP data
    dataset1_features = [
        'home_pitcher_earned_run_avg_mean', 'home_pitcher_SO_batters_faced_mean',
        'home_pitcher_H_batters_faced_mean', 'home_pitcher_BB_batters_faced_mean',
        'home_pitcher_leverage_index_avg_mean', 'home_pitcher_wpa_def_mean',
        'away_pitcher_earned_run_avg_mean', 'away_pitcher_SO_batters_faced_mean',
        'away_pitcher_H_batters_faced_mean', 'away_pitcher_BB_batters_faced_mean',
        'away_pitcher_leverage_index_avg_mean', 'away_pitcher_wpa_def_mean',
        'home_batting_efficiency', 'away_batting_efficiency',
        'home_wins_variability', 'away_wins_variability',
        'home_normalized_batting_avg', 'away_normalized_batting_avg',
        'batting_avg_interaction', 'pitching_avg_interaction', 'game_context'
    ]

    # Dataset 2: Using all pitchers (SP and relief pitchers)
    # Assuming additional pitcher stats are included; adjust accordingly
    dataset2_features = [
        # Starting Pitcher features
        'home_pitcher_earned_run_avg_mean', 'home_pitcher_SO_batters_faced_mean',
        'home_pitcher_H_batters_faced_mean', 'home_pitcher_BB_batters_faced_mean',
        'home_pitcher_leverage_index_avg_mean', 'home_pitcher_wpa_def_mean',
        'away_pitcher_earned_run_avg_mean', 'away_pitcher_SO_batters_faced_mean',
        'away_pitcher_H_batters_faced_mean', 'away_pitcher_BB_batters_faced_mean',
        'away_pitcher_leverage_index_avg_mean', 'away_pitcher_wpa_def_mean',
        # Relief Pitcher features (assuming these are present in the data)
        'home_pitching_earned_run_avg_mean', 'home_pitching_SO_batters_faced_mean',
        'home_pitching_H_batters_faced_mean', 'home_pitching_BB_batters_faced_mean',
        'home_pitching_leverage_index_avg_mean', 'home_pitching_wpa_def_mean',
        'away_pitching_earned_run_avg_mean', 'away_pitching_SO_batters_faced_mean',
        'away_pitching_H_batters_faced_mean', 'away_pitching_BB_batters_faced_mean',
        'away_pitching_leverage_index_avg_mean', 'away_pitching_wpa_def_mean',
        # Batting and other features
        'home_batting_efficiency', 'away_batting_efficiency',
        'home_wins_variability', 'away_wins_variability',
        'home_normalized_batting_avg', 'away_normalized_batting_avg',
        'batting_avg_interaction', 'pitching_avg_interaction', 'game_context'
    ]

    # Ensure that all features exist in the data
    dataset1_features = [feat for feat in dataset1_features if feat in data.columns]
    dataset2_features = [feat for feat in dataset2_features if feat in data.columns]

    # Create feature and label datasets
    X1 = data[dataset1_features]
    X2 = data[dataset2_features]
    y = data[label]

    # Feature Selection using ReliefF
    print("Performing feature selection using ReliefF for Dataset 1...")
    # Convert X1 to NumPy array
    X1_np = X1.to_numpy()
    relieff1 = ReliefF(n_features_to_select=10)
    relieff1.fit(X1_np, y)
    selected_indices1 = relieff1.top_features_[:10]
    selected_features1 = [dataset1_features[i] for i in selected_indices1]
    print(f"Selected Features for Dataset 1: {selected_features1}\n")

    print("Performing feature selection using ReliefF for Dataset 2...")
    # Convert X2 to NumPy array
    X2_np = X2.to_numpy()
    relieff2 = ReliefF(n_features_to_select=25)
    relieff2.fit(X2_np, y)
    selected_indices2 = relieff2.top_features_[:25]
    selected_features2 = [dataset2_features[i] for i in selected_indices2]
    print(f"Selected Features for Dataset 2: {selected_features2}\n")

    # Define a dictionary to hold datasets
    datasets = {
        'Dataset1_SP': (X1[selected_features1], y),
        'Dataset2_SP_RP': (X2[selected_features2], y)
    }

    # Define models to train
    models = ['ANN', '1DCNN', 'SVM']

    # To store results
    results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\nProcessing {dataset_name}...")

        # Normalize features using Min-Max Scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Handle class imbalance with SMOTE
        print("Balancing the dataset using SMOTE...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_scaled, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=RANDOM_STATE, stratify=y_res
        )

        # Initialize results storage for this dataset
        results[dataset_name] = {}

        # -----------------------------------
        # 3.1 Artificial Neural Network (ANN)
        # -----------------------------------
        print(f"\nTraining ANN for {dataset_name}...")

        # Hyperparameters to try
        ann_optimizers = ['adam', 'rmsprop']
        ann_batch_sizes = [10, 20, 30]
        ann_epochs = [50, 100, 150]

        best_ann_acc = 0
        best_ann_params = {}
        best_ann_model = None

        for optimizer in ann_optimizers:
            for batch_size in ann_batch_sizes:
                for epoch in ann_epochs:
                    print(f"ANN Training with optimizer={optimizer}, batch_size={batch_size}, epochs={epoch}")

                    # Build the model
                    ann_model = build_ann_model(input_dim=X_train.shape[1], optimizer=optimizer)

                    # Early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    # Train the model
                    history = ann_model.fit(
                        X_train, y_train,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[early_stopping]
                    )

                    # Evaluate the model
                    loss, accuracy = ann_model.evaluate(X_test, y_test, verbose=0)
                    print(f"Validation Accuracy: {accuracy:.4f}")

                    if accuracy > best_ann_acc:
                        best_ann_acc = accuracy
                        best_ann_params = {
                            'optimizer': optimizer,
                            'batch_size': batch_size,
                            'epochs': epoch
                        }
                        best_ann_model = ann_model

        print(f"Best ANN Params for {dataset_name}: {best_ann_params}")
        print(f"Best ANN Test Accuracy: {best_ann_acc:.4f}")

        # Store ANN results
        results[dataset_name]['ANN'] = {
            'model': best_ann_model,
            'accuracy': best_ann_acc,
            'params': best_ann_params
        }

        # Evaluate and store metrics
        y_pred_proba = best_ann_model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        print("ANN Evaluation Metrics:")
        acc, conf_matrix, class_report, roc_auc = evaluate_model(y_test, y_pred, y_pred_proba)
        results[dataset_name]['ANN'].update({
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc
        })

        # -----------------------------------
        # 3.2 One-Dimensional Convolutional Neural Network (1DCNN)
        # -----------------------------------
        print(f"\nTraining 1DCNN for {dataset_name}...")

        # Reshape data for 1DCNN: samples, timesteps, features
        # Since data is tabular, we can treat each feature as a timestep with 1 feature
        X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Hyperparameters to try
        cnn_optimizers = ['adam', 'rmsprop']
        cnn_batch_sizes = [10, 20, 30]
        cnn_epochs = [50, 100, 150]

        best_cnn_acc = 0
        best_cnn_params = {}
        best_cnn_model = None

        for optimizer in cnn_optimizers:
            for batch_size in cnn_batch_sizes:
                for epoch in cnn_epochs:
                    print(f"1DCNN Training with optimizer={optimizer}, batch_size={batch_size}, epochs={epoch}")

                    # Build the model
                    cnn_model = build_1dcnn_model(input_shape=(X_train_cnn.shape[1], 1), optimizer=optimizer)

                    # Early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    # Train the model
                    history = cnn_model.fit(
                        X_train_cnn, y_train,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[early_stopping]
                    )

                    # Evaluate the model
                    loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
                    print(f"Validation Accuracy: {accuracy:.4f}")

                    if accuracy > best_cnn_acc:
                        best_cnn_acc = accuracy
                        best_cnn_params = {
                            'optimizer': optimizer,
                            'batch_size': batch_size,
                            'epochs': epoch
                        }
                        best_cnn_model = cnn_model

        print(f"Best 1DCNN Params for {dataset_name}: {best_cnn_params}")
        print(f"Best 1DCNN Test Accuracy: {best_cnn_acc:.4f}")

        # Store 1DCNN results
        results[dataset_name]['1DCNN'] = {
            'model': best_cnn_model,
            'accuracy': best_cnn_acc,
            'params': best_cnn_params
        }

        # Evaluate and store metrics
        y_pred_proba = best_cnn_model.predict(X_test_cnn).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        print("1DCNN Evaluation Metrics:")
        acc, conf_matrix, class_report, roc_auc = evaluate_model(y_test, y_pred, y_pred_proba)
        results[dataset_name]['1DCNN'].update({
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc
        })

        # -----------------------------------
        # 3.3 Support Vector Machine (SVM)
        # -----------------------------------
        print(f"\nTraining SVM for {dataset_name}...")

        # Define the SVM pipeline
        svm_pipeline = Pipeline([
            ('svc', SVC(probability=True, random_state=RANDOM_STATE))
        ])

        # Define parameter grid
        param_grid = {
            'svc__kernel': ['linear', 'rbf'],
            'svc__C': [1, 10, 100, 1000],
            'svc__gamma': [0.0001, 0.001, 0.1, 1, 10, 100]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=svm_pipeline,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=0
        )

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        print(f"Best SVM Params for {dataset_name}: {grid_search.best_params_}")
        print(f"Best SVM Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

        # Best estimator
        best_svm = grid_search.best_estimator_

        # Evaluate on test set
        y_pred_proba = best_svm.predict_proba(X_test)[:, 1]
        y_pred = best_svm.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"SVM Test Accuracy: {acc:.4f}")

        # Store SVM results
        results[dataset_name]['SVM'] = {
            'model': best_svm,
            'accuracy': acc,
            'params': grid_search.best_params_
        }

        # Evaluate and store metrics
        print("SVM Evaluation Metrics:")
        acc, conf_matrix, class_report, roc_auc = evaluate_model(y_test, y_pred, y_pred_proba)
        results[dataset_name]['SVM'].update({
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc
        })

    # 6. Summary of Results
    def summarize_results(results):
        print("\n\nSummary of Results:")
        for dataset_name, model_results in results.items():
            print(f"\n{dataset_name}:")
            for model_name, metrics in model_results.items():
                print(f"  {model_name} Accuracy: {metrics['accuracy']:.4f}")

        # Optional: Plotting
        datasets_names = list(results.keys())
        ann_accuracies = [results[ds]['ANN']['accuracy'] for ds in datasets_names]
        cnn_accuracies = [results[ds]['1DCNN']['accuracy'] for ds in datasets_names]
        svm_accuracies = [results[ds]['SVM']['accuracy'] for ds in datasets_names]

        x = np.arange(len(datasets_names))
        width = 0.2

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, ann_accuracies, width, label='ANN')
        plt.bar(x, cnn_accuracies, width, label='1DCNN')
        plt.bar(x + width, svm_accuracies, width, label='SVM')

        plt.xlabel('Datasets')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracies on Different Datasets')
        plt.xticks(x, datasets_names)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
    # After main(), call summarize_results if needed
    # summarize_results(results)  # Uncomment if results are accessible here

