#!/usr/bin/env python3
"""
Script to extract data from the third sheet 'Indices in EUR' from the Excel file
and convert it to CSV format.
"""

import pandas as pd
import sys
import os

def extract_indices_to_csv(excel_file_path, output_csv_path=None):
    """
    Extract data from the third sheet 'Indices in EUR' and save as CSV.
    
    Args:
        excel_file_path (str): Path to the Excel file
        output_csv_path (str): Path for the output CSV file (optional)
    """
    try:
        # Read the Excel file and get all sheet names
        excel_file = pd.ExcelFile(excel_file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"Found {len(sheet_names)} sheets in the Excel file:")
        for i, sheet_name in enumerate(sheet_names, 1):
            print(f"  {i}. {sheet_name}")
        
        # Check if we have at least 3 sheets
        # if len(sheet_names) < 3:
        #     print(f"Error: The Excel file only has {len(sheet_names)} sheets, but we need at least 3.")
        #     return False
        
        # Find the "Indices in EUR" sheet
        target_sheet = "Indices in EUR"
        if target_sheet in sheet_names:
            sheet_name = target_sheet
            sheet_index = sheet_names.index(target_sheet) + 1
            print(f"\nFound '{target_sheet}' sheet (sheet #{sheet_index})")
        else:
            print(f"\nError: Sheet '{target_sheet}' not found in the Excel file.")
            print(f"Available sheets: {sheet_names}")
            return False
        
        # Read the target sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        
        # Display basic info about the data
        print(f"\nData shape: {df.shape} (rows x columns)")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Set default output path if not provided
        if output_csv_path is None:
            output_csv_path = "indices_eur.csv"
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nData successfully saved to: {output_csv_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file_path}' not found.")
        return False
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return False

def main():
    # Set the Excel file path
    excel_file = "indices final.xlsx"
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"Error: Excel file '{excel_file}' not found in current directory.")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Extract data to CSV
    success = extract_indices_to_csv(excel_file)
    
    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed!")

if __name__ == "__main__":
    main()
