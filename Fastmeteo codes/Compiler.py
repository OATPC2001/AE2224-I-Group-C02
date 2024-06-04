import pandas as pd

def compile_data():

    # Add the file paths of the data files to be compiled if the data is broken into multiple files
    file_paths = [
        
    ]

    # Initialize an empty list to store dataframes
    dfs = []

    # Read each file and append its dataframe to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all dataframes into a single dataframe
    compiled_df = pd.concat(dfs, ignore_index=True)

    # Save the compiled dataframe to a CSV file
    file_path = 'Generated data\\20--\\20--_Wind_velocities.csv' # Change the year in the file name

    compiled_df.to_csv(file_path, index=False)

    print(f"Data has been compiled and saved to {file_path}.")


def decompiler():

    # Read the compiled data
    file_path = '20--\\20--_Wind_velocities.csv'
    df = pd.read_csv(file_path)

    # Split the data into separate dataframes
    dfs = []
    for i in range(9):
        start = i * 1000
        end = (i + 1) * 1000
        dfs.append(df.iloc[start:end])

    # Save each dataframe to a separate CSV file
    for i, df in enumerate(dfs):
        file_path = f'20--\\20--_Wind_velocities_{i + 1}.csv' # Change the year in the file name
        df.to_csv(file_path, index=False)

    print(f"Data has been decompiled and saved to 20--\\20--_Wind_velocities_[1-9].csv.") # Change the year in the file name