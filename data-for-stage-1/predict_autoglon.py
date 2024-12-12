import pandas as pd
from autogluon.tabular import TabularPredictor
from datetime import datetime

# File paths
model_dir = "/Users/chuajerome/AutogluonModels/ag-20241212_050536"  # Replace with your model directory path
test_file_path = "/Users/chuajerome/Desktop/ML_Project/data-for-stage-1/same_season_test_data.csv"

# Generate a datetime-based filename
current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")  # Format: YYYYMMDDHHMMSS
output_file_path = f"/Users/chuajerome/Desktop/ML_Project/data-for-stage-1/predictions_{current_datetime}.csv"

# Load the trained predictor
predictor = TabularPredictor.load(model_dir)

# Load the test dataset
test_df = pd.read_csv(test_file_path)

# Ensure 'id' exists in the test dataset
if 'id' not in test_df.columns:
    raise ValueError("'id' column is missing in the test dataset.")

# Drop unnecessary columns in the test dataset
if 'date' in test_df.columns:
    test_df = test_df.drop(columns=['date'])

# Make predictions
predictions = predictor.predict(test_df)

# Create an output DataFrame
output_df = pd.DataFrame({
    "id": test_df["id"],
    predictor.label: predictions
})

# Save predictions to a dynamically named CSV file
output_df.to_csv(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")
