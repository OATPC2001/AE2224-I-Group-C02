import pandas as pd
import os

# Assuming all CSV files are in the current directory
directory = '.'
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# List to store all DataFrames
dfs = []

# Read each CSV file and append its DataFrame to the list
for file_name in csv_files:
    df = pd.read_csv(file_name)
    dfs.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)

# Convert the date column to datetime if it's not already
merged_df['Actual Off Block Date'] = pd.to_datetime(merged_df['Actual Off Block Date'], format='%d-%m-%Y')

# Sort the DataFrame by the date column in descending order (latest to most recent)
merged_df.sort_values(by='Actual Off Block Date', ascending=True, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data_ordered_emissions.csv', index=False)