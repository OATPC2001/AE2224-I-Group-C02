import pandas as pd
import numpy as np

# Read the flight points data from the specified file
flight_points = pd.read_csv('Python files\Flight_Points_Actual_20190301_20190331.csv.gz')

# Set display options for pandas and numpy
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=30)

# Sort the flight points data by 'ECTRL ID' and 'Sequence Number'
flight_points = flight_points.sort_values(by=['ECTRL ID', 'Sequence Number'])

# Convert 'Flight Level' from feet to meters
flight_points['Flight Level'] *= 30.48

# Remove duplicate entries based on 'ECTRL ID' and 'Time Over', keeping the last occurrence
flight_points.drop_duplicates(subset=['ECTRL ID', 'Time Over'], keep='last', inplace=True)


def haversine_distance(df):
    # Convert latitude and longitude to radians
    df[['Latitude', 'Longitude']] = np.deg2rad(df[['Latitude', 'Longitude']])

    # Calculate the difference in latitude and longitude for each 'ECTRL ID'
    grouped_df = df.groupby('ECTRL ID')[['Latitude', 'Longitude']]
    dlat, dlon = grouped_df.diff().T.values

    # Calculate haversine distance
    a = np.sin(dlat / 2.0) ** 2 + np.cos(df['Latitude'].shift(-1)) * np.cos(df['Latitude']) * np.sin(dlon / 2.0) ** 2
    distances = 6371 * 1000 * 2 * np.arcsin(np.sqrt(a))

    return distances


# Convert 'Time Over' column to datetime format
flight_points['Time Over'] = pd.to_datetime(flight_points['Time Over'], format='%d-%m-%Y %H:%M:%S')

# Calculate the time difference in seconds for each 'ECTRL ID'
flight_points['time_delta(s)'] = flight_points.groupby('ECTRL ID')['Time Over'].diff().dt.total_seconds()

# Calculate haversine distance for each point
flight_points['dist(m)'] = pd.Series(haversine_distance(flight_points))

# Calculate speed in meters per second
flight_points['speed(m/s)'] = flight_points['dist(m)'] / flight_points['time_delta(s)']

# Calculate the change in altitude for each 'ECTRL ID'
flight_points['altitude_delta'] = flight_points.groupby('ECTRL ID')[['Flight Level']].diff()

# Calculate the flight angle in radians
flight_points['flight_angle(rad)'] = np.arctan(np.divide(flight_points['altitude_delta'], flight_points['dist(m)'],
                                                         out=np.zeros_like(flight_points['altitude_delta']),
                                                         where=flight_points['dist(m)'] != 0))

# Select the final columns and rename them
final_df = flight_points[['ECTRL ID', 'speed(m/s)', 'dist(m)', 'Latitude', 'Longitude', 'Flight Level', 'Time Over',
                          'time_delta(s)', 'flight_angle(rad)']].rename(
    columns={'Flight Level': 'altitude(m)', 'Time Over': 'timestamp'})

# Save the final dataframe to a CSV file
final_df.to_csv('Flight_points_with_speed_2019.csv', sep=',', index=False, encoding='utf-8')
