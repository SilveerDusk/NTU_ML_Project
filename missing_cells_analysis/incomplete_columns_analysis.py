"""
Description: Analysis of empty cells for every column
"""

import pandas as pd

def for_column_name_find_missing_rows(data):
    # Initialize a dictionary to store columns with their respective IDs that have empty cells
    column_name_to_empty_row_ids = {}

    # Loop through each column to find rows with missing values
    for column in data.columns:
        # Find rows where the current column has missing values
        ids_with_empty_in_column = data[data[column].isnull()]['id'].tolist()

        # If there are any IDs with missing values in this column, add them to the dictionary
        column_name_to_empty_row_ids[column] = ids_with_empty_in_column

    # Output the result
    print("Columns with missing values and their respective IDs:")
    for column, ids in column_name_to_empty_row_ids.items():
        print(f"Column: {column}, IDs with empty values: {ids}")

    return column_name_to_empty_row_ids

def for_column_name_count_number_of_missing_rows(column_name_to_empty_row_ids):
    column_name_to_number_of_empty_rows = {}
    for column_name, empty_rows in column_name_to_empty_row_ids.items():
        column_name_to_number_of_empty_rows[column_name] = len(empty_rows)
    return column_name_to_number_of_empty_rows

def main():
    file_path = '../data-for-stage-1/train_data.csv'
    data = pd.read_csv(file_path)
    column_name_to_empty_row_ids = for_column_name_find_missing_rows(data)
    column_name_to_number_of_empty_rows = for_column_name_count_number_of_missing_rows(column_name_to_empty_row_ids)
    print(column_name_to_number_of_empty_rows)
    pass

if __name__ == '__main__':
    main()