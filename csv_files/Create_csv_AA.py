# Create CSV file from our data:
import pandas as pd

# Path to your Excel file
excel_file = "csv_files/long_factor_regression_data.xlsx"

# List of sheet names (arcs) you want to export
sheets_to_export = ["EXPORT US USD", "EXPORT US EUR", "EXPORT EU EUR"]

# Loop through and save each as CSV
for sheet in sheets_to_export:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    df.to_csv(f"csv_files/long_{sheet}.csv", index=False)
    print(f"Saved {sheet}.csv")