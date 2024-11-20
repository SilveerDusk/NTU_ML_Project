"""
Definition:
- display name: Original name from Columns Overview
- internal name: display name converted to snake case (lower case and spaces replaced with underscore) and without text in brackets

Description
- Get internal category name for column name
- Get column names for internal category name
- Get list of all internal category names
"""

import json

# Initialize categories to None, so it only loads when needed
categories = None


def load_categories(file_path="column_categories.json"):
    """Loads categories from JSON file if not already loaded. Returns categories dictionary.
       If the file is missing or an error occurs, initializes categories to an empty dictionary.
    """
    global categories
    if categories is None:  # Only load if categories have not been loaded yet
        try:
            with open(file_path, "r") as file:
                categories = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading file '{file_path}': {e}")
            categories = {}  # Initialize to empty to avoid further errors
    return categories


def get_category_for_column(column_name):
    """Returns the internal category name for a given column, or 'Unknown Category' if not found."""
    load_categories()  # Ensure categories are loaded
    for category, data in categories.items():
        if column_name in data["columns"]:
            return category
    return "Unknown Category"


def get_columns_for_category(category):
    """Returns a list of columns for a given internal category name, or an empty list if the category does not exist."""
    load_categories()  # Ensure categories are loaded
    return categories.get(category, {}).get("columns", [])


def get_all_categories():
    """Returns a list of all internal category names."""
    load_categories()  # Ensure categories are loaded
    return list(categories.keys())


def main():
    # Example usage
    print(get_columns_for_category("game_information"))  # Returns all columns under "game_information"
    print(get_category_for_column("home_team_abbr"))  # Should return "game_information"
    print(get_all_categories())  # Should return all internal category names


if __name__ == '__main__':
    main()
