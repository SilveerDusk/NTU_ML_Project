from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm


def split_data(y_train, X_train, sub_training_size=8000):
    X_sub_train, X_rest_train, y_sub_train, y_rest_train = train_test_split(
        X_train, y_train, train_size=sub_training_size, random_state=42
    )
    return X_sub_train, y_sub_train, X_rest_train, y_rest_train


def csv_to_numpy(csv_path, select_columns):
    df = pd.read_csv(csv_path)

    # Ensure the selected columns are in the DataFrame
    df = df[select_columns + ['home_team_win']]

    # Convert 'home_team_win' to binary values (if not already)
    labels = df['home_team_win'].astype(int)

    # Keep only numeric columns for features
    features = df[select_columns].apply(pd.to_numeric, errors='coerce')

    # Replace missing values with column means
    features = features.fillna(features.mean())

    # Scale data for SVM
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels



def calculate_error(model, X, y):
    y_pred = model.predict(X)
    error = 1 - accuracy_score(y, y_pred)
    return error * 100  # Convert to percentage


def select_best_lambda_star(X, y, C_values=[0.01, 0.1, 1, 10, 100]):
    best_C = None
    smallest_error = float('inf')
    best_model = None

    for C in C_values:
        model = SVC(C=C, kernel='rbf', random_state=42)  # Use RBF kernel for non-linear data
        model.fit(X, y)
        error = calculate_error(model, X, y)

        if error < smallest_error:
            smallest_error = error
            best_C = C
            best_model = model

    return best_C, best_model


def repeat_experiment(X_train, y_train, number_of_repetitions=2, sub_training_size=8000):
    error_outs = []

    for _ in tqdm(range(number_of_repetitions)):
        X_sub_train, y_sub_train, X_rest_train, y_rest_train = split_data(
            y_train, X_train, sub_training_size
        )
        best_C, model = select_best_lambda_star(X_sub_train, y_sub_train)
        error_out = calculate_error(model, X_rest_train, y_rest_train)
        error_outs.append(error_out)

    save_data_to_file(error_outs, "e_outs")
    return error_outs


def save_data_to_file(data, filename_prefix=""):
    folder_name = "output_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder_name, f"{filename_prefix}_{timestamp}.txt")
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")
    print(f"Data saved to {filename}")


def plot_histogram_eout(e_outs, problem=""):
    plt.hist(e_outs, bins=30)
    plt.xlabel("E_out(g)")
    plt.ylabel("Frequency")
    plt.title("Histogram of E_out over experiments")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{problem}-histogram_eout_{timestamp}.png")
    plt.show()


def main():
    training_file = "../data-for-stage-1/train_data.csv"
    columns = [
        "away_pitching_SO_batters_faced_10RA",
        "home_pitching_earned_run_avg_mean",
        "away_pitcher_SO_batters_faced_mean",
        "home_batting_onbase_plus_slugging_mean",
        "home_pitching_H_batters_faced_10RA",
        "home_pitching_SO_batters_faced_10RA",
        "away_pitching_SO_batters_faced_mean",
        "home_pitching_BB_batters_faced_mean",
        "home_batting_onbase_plus_slugging_skew",
        "away_pitching_leverage_index_avg_mean"
    ]

    X_train, y_train = csv_to_numpy(training_file, columns)
    e_outs = repeat_experiment(X_train, y_train)
    print(e_outs)
    print(np.mean(e_outs))
    plot_histogram_eout(e_outs, "")


if __name__ == '__main__':
    main()
