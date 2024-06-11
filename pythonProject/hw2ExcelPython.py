import pandas as pd
from openpyxl import Workbook
from pathlib import Path

# Define file paths
logs_path = Path("logs")
BCM_file = "BMC.xlsx" # Assigning string to variable

# Creates a new workbook and by default the new workbook contains a single sheet
wb = Workbook()
wb.save(BCM_file)


# Create excel writer object to write to BCM.xlsx
with pd.ExcelWriter(BCM_file, mode="a", engine="openpyxl") as writer:
    for file in logs_path.glob("*.csv"):
        try:
            # Read csv files into pandas dataframe
            df = pd.read_csv(file)

            # Get filename to use as sheet name
            sheet_name = file.stem

            # Insert dataframe into excel file as new sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Inserted {file.name} into {BCM_file} as {sheet_name} sheet.")
        except Exception as e:
            print(f"Failed to insert {file.name}: {e}")














