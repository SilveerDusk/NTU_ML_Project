"""
Description: Analysis of empty cells for every row
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
"""
Helper functions
"""


def rows_to_empty_column_names(data):
    # Initialize a dictionary to store IDs and their respective empty columns
    id_to_empty_column_names = {}

    # Iterate through each row to check for missing values
    for index, row in data.iterrows():
        # Identify columns with missing values in the current row
        empty_columns = row[row.isnull()].index.tolist()

        # Add the ID and empty columns to the dictionary
        id_to_empty_column_names[row['id']] = empty_columns

    return id_to_empty_column_names


def count_missing_columns(id_to_empty_column_list):
    id_to_number_of_empty_columns = {}
    for id, empty_cols in id_to_empty_column_list.items():
        id_to_number_of_empty_columns[id] = len(empty_cols)
    return id_to_number_of_empty_columns


def separate_by_home_team_win(data, id_to_number_empty_columns):
    # Initialize dictionaries to store IDs based on the outcome of home team win
    home_team_win_dict = {}
    home_team_loss_dict = {}

    # Iterate over each ID and check if the home team won
    for id, count in id_to_number_empty_columns.items():
        # Retrieve the row corresponding to the current ID
        home_team_won = data.loc[data['id'] == id, 'home_team_win'].values[0]

        # Separate IDs based on whether home team won or not
        if home_team_won:
            home_team_win_dict[id] = count
        else:
            home_team_loss_dict[id] = count

    return home_team_win_dict, home_team_loss_dict


def cumulative_count(data_dict, num_indices):
    # Sort the dictionary by keys to ensure ordered summation
    sorted_items = sorted(data_dict.items())

    # Initialize a list of size `num_indices` with all zeros
    cumulative_list = [0] * num_indices

    # Initialize the cumulative sum
    cumulative_sum = 0

    # Iterate through the sorted dictionary and compute the cumulative sum
    for index, value in sorted_items:
        cumulative_sum += value
        # Update the cumulative_list at the current index
        if index < num_indices:
            cumulative_list[index] = cumulative_sum

    # Fill any missing values in cumulative_list with the last cumulative sum
    for i in range(1, num_indices):
        if cumulative_list[i] == 0:
            cumulative_list[i] = cumulative_list[i - 1]

    return cumulative_list


def convert_to_percentage(cumulative_list, as_decimal=False):
    # Get the final cumulative value (the last element in the list)
    total = cumulative_list[-1]

    # Avoid division by zero in case the total is zero
    if total == 0:
        raise ValueError("Total cumulative sum is zero, cannot calculate percentages.")

    # Convert each element to a percentage of the total
    percentage_list = [(value / total) * (1.0 if as_decimal else 100) for value in cumulative_list]

    return percentage_list


"""
Plots
"""

def calculate_proportion_home_win_to_loss(home_team_win_dict, home_team_loss_dict, bin_size=10):
    # Determine the maximum number of missing elements to create bins
    max_count = max(max(home_team_win_dict.values()), max(home_team_loss_dict.values()))
    bins = range(0, max_count + bin_size, bin_size)

    # Calculate bin counts for both win and loss dictionaries
    win_counts, _ = np.histogram(list(home_team_win_dict.values()), bins=bins)
    loss_counts, _ = np.histogram(list(home_team_loss_dict.values()), bins=bins)

    # Calculate proportion as win_count / loss_count, handling division by zero
    proportions = np.divide(win_counts, loss_counts, out=np.zeros_like(win_counts, dtype=float), where=loss_counts != 0)

    # Plot the proportions
    plt.figure(figsize=(10, 6))
    plt.plot(bins[:-1], proportions, marker='o', color='purple', label="Home Win to Loss Proportion")
    plt.xlabel("Number of Missing Elements (Binned)")
    plt.ylabel("Win to Loss Proportion")
    plt.title("Proportion of Home Team Win to Loss by Missing Elements")
    plt.legend()

    # Focus on the important range (where proportion is not zero)
    plt.ylim(0, max(proportions[proportions > 0]) * 1.1)  # Zoom to focus on non-zero values
    plt.show()


def plot_missing_columns_histogram(id_to_number_empty_columns, bin_size=10):
    # Extract the number of missing elements
    missing_counts = list(id_to_number_empty_columns.values())

    # Determine the range of bins based on the specified bin size
    max_count = max(missing_counts)
    bins = range(0, max_count + bin_size, bin_size)

    # Plot the histogram with custom bins
    plt.figure(figsize=(10, 6))
    plt.hist(missing_counts, bins=bins, edgecolor='black', align='left')
    plt.xlabel("Number of Missing Elements (Binned)")
    plt.ylabel("Frequency")
    plt.title("Binned Histogram of Missing Elements Frequency")
    plt.xticks(bins)  # Set x-ticks to match the bins
    plt.show()


def for_row_plot_binned_missing_elements_line_dict(id_dicts, bin_size=10, labels=None):
    plt.figure(figsize=(10, 6))
    max_count = 0  # Track the max count for setting bins

    # Loop over each dictionary to calculate and plot normalized percentages
    for i, id_dict in enumerate(id_dicts):
        # Extract the number of missing elements
        missing_counts = list(id_dict.values())

        # Update max_count if current max missing value count is higher
        max_count = max(max_count, max(missing_counts))

        # Calculate frequency counts in specified bins
        bins = range(0, max_count + bin_size, bin_size)
        bin_counts, _ = np.histogram(missing_counts, bins=bins)

        # Normalize to percentage
        bin_percentages = (bin_counts / bin_counts.sum()) * 100

        # Set label if provided, otherwise use a generic label
        label = labels[i] if labels else f"Dataset {i + 1}"

        # Plot line graph of bin percentages
        plt.plot(bins[:-1], bin_percentages, marker='o', label=label)

    # Adding labels and title
    plt.xlabel("Number of Missing Elements (Binned)")
    plt.ylabel("Percentage")
    plt.title("Percentage of Missing Elements by Outcome")
    plt.legend()
    plt.show()

def plot_cumulative_percentages(cumulative_percentage_home_team_win, cumulative_percentage_home_team_loss):
    # Define the x-axis values as indices
    indices = list(range(len(cumulative_percentage_home_team_win)))

    # Plot both lists
    plt.figure(figsize=(10, 6))
    plt.plot(indices, cumulative_percentage_home_team_win, label="Home Team Win", color='blue', marker='o')
    plt.plot(indices, cumulative_percentage_home_team_loss, label="Home Team Loss", color='red', marker='o')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Cumulative Percentage")
    plt.title("Cumulative Percentage of Home Team Win and Loss")
    plt.legend()

    # Set y-axis limits from 0 to 100%
    plt.ylim(0, 100)

    plt.show()

def plot_cumulative_difference(cumulative_percentage_win, cumulative_percentage_loss):
    # Calculate the difference between cumulative win and loss percentages
    cumulative_difference = [win - loss for win, loss in zip(cumulative_percentage_win, cumulative_percentage_loss)]

    # Define the x-axis values as indices
    indices = list(range(len(cumulative_difference)))

    # Plot the difference
    plt.figure(figsize=(10, 6))
    plt.plot(indices, cumulative_difference, color='purple', marker='o', linestyle='-', label="Difference (Win - Loss)")

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Cumulative Percentage Difference")
    plt.title("Difference in Cumulative Percentage: Home Team Win - Home Team Loss")
    plt.axhline(0, color='gray', linestyle='--')  # Add a horizontal line at y=0 for reference
    plt.legend()

    # Show the plot
    plt.show()

def plot_growth_rate(cumulative_percentage_win, cumulative_percentage_loss):
    # Calculate the growth rate (derivative) as the difference between consecutive points
    growth_rate_win = [cumulative_percentage_win[i] - cumulative_percentage_win[i - 1] for i in range(1, len(cumulative_percentage_win))]
    growth_rate_loss = [cumulative_percentage_loss[i] - cumulative_percentage_loss[i - 1] for i in range(1, len(cumulative_percentage_loss))]

    # Define the x-axis values as indices (starting from 1 due to derivative calculation)
    indices = list(range(1, len(cumulative_percentage_win)))

    # Plot the growth rates for both win and loss
    plt.figure(figsize=(10, 6))
    plt.plot(indices, growth_rate_win, label="Home Team Win Growth Rate", color='blue')
    plt.plot(indices, growth_rate_loss, label="Home Team Loss Growth Rate", color='red')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Growth Rate")
    plt.title("Growth Rate of Cumulative Percentage for Home Team Win and Loss")
    plt.legend()

    # Show the plot
    plt.show()

def plot_smoothed_growth_rate(cumulative_percentage_win, cumulative_percentage_loss, window_size=50):
    # Calculate the growth rate (derivative) as the difference between consecutive points
    growth_rate_win = [cumulative_percentage_win[i] - cumulative_percentage_win[i - 1] for i in range(1, len(cumulative_percentage_win))]
    growth_rate_loss = [cumulative_percentage_loss[i] - cumulative_percentage_loss[i - 1] for i in range(1, len(cumulative_percentage_loss))]

    # Convert lists to pandas Series for easier moving average calculation
    growth_rate_win_series = pd.Series(growth_rate_win)
    growth_rate_loss_series = pd.Series(growth_rate_loss)

    # Apply a rolling window to calculate the moving average
    smoothed_growth_rate_win = growth_rate_win_series.rolling(window=window_size).mean()
    smoothed_growth_rate_loss = growth_rate_loss_series.rolling(window=window_size).mean()

    # Define the x-axis values as indices (starting from 1 due to derivative calculation)
    indices = list(range(1, len(cumulative_percentage_win)))

    # Plot the smoothed growth rates for both win and loss
    plt.figure(figsize=(10, 6))
    plt.plot(indices, smoothed_growth_rate_win, label="Smoothed Home Team Win Growth Rate", color='blue')
    plt.plot(indices, smoothed_growth_rate_loss, label="Smoothed Home Team Loss Growth Rate", color='red')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Smoothed Growth Rate")
    plt.title("Smoothed Growth Rate of Cumulative Percentage for Home Team Win and Loss")
    plt.legend()

    # Show the plot
    plt.show()

def plot_missing_elements_absolute_with_index(data):
    # Initialize lists to store (index, count) pairs for each outcome
    home_team_win_counts = []
    home_team_loss_counts = []

    # Iterate over each row and store the actual index and count of empty elements
    for index, row in data.iterrows():
        empty_count = row.isnull().sum()  # Count of empty elements in the row
        if row['home_team_win']:
            home_team_win_counts.append((index, empty_count))
        else:
            home_team_loss_counts.append((index, empty_count))

    # Separate indices and counts for plotting
    win_indices, win_counts = zip(*home_team_win_counts)
    loss_indices, loss_counts = zip(*home_team_loss_counts)

    # Plotting with transparency to allow overlap blending
    plt.figure(figsize=(14, 7))
    plt.scatter(win_indices, win_counts, color='blue', label="Home Team Win", alpha=0.5, marker='o')
    plt.scatter(loss_indices, loss_counts, color='red', label="Home Team Loss", alpha=0.5, marker='o')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Number of Empty Elements")
    plt.title("Number of Empty Elements per Row by Home Team Win and Loss (Overlap Blending)")
    plt.legend()

    # Show the plot
    plt.show()

def plot_mixed_overlay_missing_elements(data):
    # Initialize lists to store the x-values (indices), y-values (empty counts), and colors
    indices = []
    empty_counts = []
    colors = []

    # Iterate over each row to populate the indices, counts, and colors
    for index, row in data.iterrows():
        empty_count = row.isnull().sum()  # Count of empty elements in the row
        indices.append(index)
        empty_counts.append(empty_count)
        # Set color based on outcome: blue for win, red for loss
        if row['home_team_win']:
            colors.append((0, 0, 1, 0.5))  # Blue with alpha for transparency
        else:
            colors.append((1, 0, 0, 0.5))  # Red with alpha for transparency

    # Plot all points in a single scatter plot, with colors that will blend where they overlap
    plt.figure(figsize=(14, 7))
    plt.scatter(indices, empty_counts, color=colors, marker='o')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Number of Empty Elements")
    plt.title("Number of Empty Elements per Row by Home Team Win and Loss (Mixed Overlay)")
    plt.legend(["Blue = Win, Red = Loss, Purple = Overlap"], loc="upper left")

    # Show the plot
    plt.show()

def plot_density_map_missing_elements(data, bins=(100, 100)):
    # Initialize lists to store the indices and number of empty elements for all points
    indices = []
    empty_counts = []

    # Iterate over each row and collect the indices and empty counts
    for index, row in data.iterrows():
        empty_count = row.isnull().sum()  # Count of empty elements in the row
        indices.append(index)
        empty_counts.append(empty_count)

    # Create a 2D histogram to visualize density
    plt.figure(figsize=(14, 7))
    plt.hist2d(indices, empty_counts, bins=bins, cmap='plasma', density=True)

    # Add color bar to indicate density levels
    plt.colorbar(label='Density')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Number of Empty Elements")
    plt.title("Density Map of Number of Empty Elements per Row by Index")

    # Show the plot
    plt.show()

"""
Main
"""
def main():
    file_path = '../data-for-stage-1/train_data.csv'
    data = pd.read_csv(file_path)
    num_indices = data.shape[0]
    id_to_empty_columns = rows_to_empty_column_names(data)
    id_to_number_empty_columns = count_missing_columns(id_to_empty_columns)

    # Separate IDs by home team win status
    home_team_win_dict, home_team_loss_dict = separate_by_home_team_win(data, id_to_number_empty_columns)

    # Print separated dictionaries
    print("Home Team Win Dict:", home_team_win_dict)
    print("Home Team Loss Dict:", home_team_loss_dict)

    cumulative_home_team_win = cumulative_count(home_team_win_dict, num_indices)
    cumulative_home_team_loss = cumulative_count(home_team_loss_dict, num_indices)

    cumulative_percentage_home_team_win = convert_to_percentage(cumulative_home_team_win)
    cumulative_percentage_home_team_loss = convert_to_percentage(cumulative_home_team_loss)

if __name__ == '__main__':
    main()