# This imports the workbook class from openpyxl
from openpyxl import Workbook
import os

# Creates a new workbook and by default the new workbook contains a single sheet
wb = Workbook()

# Save workbook
filename = "BCM.xlsx"
wb.save(filename)

# Check if file was created
if os.path.exists(filename):
    print(f"File {filename} exists")
else:
    print(f"File {filename} does not exist")



