import pandas as pd

# Specify the file name
file_name = 'CO2_long_haul_svm.csv'

# Read the specified CSV file
df = pd.read_csv(file_name)

# Convert the date column to datetime specifying the current format
df['Actual Off Block Date'] = pd.to_datetime(df['Actual Off Block Date'], format='%d-%m-%Y')

# Format the date column to day month year
df['Actual Off Block Date'] = df['Actual Off Block Date'].dt.strftime('%Y-%m-%d')

# Save the DataFrame to a new CSV file
df.to_csv( file_name, index=False)

#can change formats according to need