import numpy as np
import pandas as pd
import openap

pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 15)

# Load the data
wind_v = pd.read_csv('windvtest.csv')
df_flights = pd.read_csv('masses16_2.csv')

# Add a row identifier to each DataFrame
wind_v['row_id'] = wind_v.groupby('ECTRL ID').cumcount()
df_flights['row_id'] = df_flights.groupby('ECTRL ID').cumcount()

# Merge on both ECTRL ID and row_id
merged_df = pd.merge(df_flights, wind_v, on=['ECTRL ID', 'row_id'], how='left').fillna(0)

# Print the result
print(merged_df)

# Calculate the true speed
merged_df['true speed'] = merged_df['speed(m/s)'] + merged_df['delta_v']

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_df.csv', sep=',', index=False, encoding='utf-8')
