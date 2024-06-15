import pandas as pd

df = pd.read_csv('filtered_flights_final.csv')

df['Actual Off Block Date'] = pd.to_datetime(df['Actual Off Block Date'], format='%Y-%m-%d')
df_sorted = df.sort_values(by='Actual Off Block Date',ascending = True)

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('filtered_Flights_ordered.csv', index=False)

