import pandas as pd
import os
from openpyxl import Workbook
from glob import glob
from pathlib import Path
import shutil

# Analysis 2 – Wrangling csv files
# Get current working directory
cwd = os.getcwd()
print(cwd)
print(type(cwd))

# List all files and directories in data directory
print(os.listdir("./logs"))

# Create a Path object corresponding to the new directory name
logs_dir = Path.cwd() / 'logs'
print(logs_dir)

# Make the new directory
logs_dir.mkdir(exist_ok=True)
print(logs_dir.is_dir())

# Find all csv files in logs directory
csv_files = list(logs_dir.glob('*.csv'))

# Create a Pandas Excel writer using openpyxl as the engine
BCM_file = logs_dir / 'BCM.xlsx'
with pd.ExcelWriter(BCM_file, engine='openpyxl') as writer:
    for csv_file in csv_files:
        # Read each CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Add headers
        headers = ['Datetime', 'Scale', 'Temperature']
        df.columns = headers

        # Use the name of the csv file as the sheet name
        sheet_name = csv_file.stem

        # Write the DataFrame to an Excel sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"All CSV files have been copied to {BCM_file}")













