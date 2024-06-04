import pandas as pd
import numpy as np

# change this file into the year required
flight_points = pd.read_csv('General data\Flight_Points_Actual_20190301_20190331.csv.gz') # Add Flight Points Actual file for the month and year of the data
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=30)

flight_points = flight_points.sort_values(by=['ECTRL ID', 'Sequence Number'])
flight_points['Flight Level'] *= 30.48

flight_points.drop_duplicates(subset=['ECTRL ID', 'Time Over'], keep='last', inplace=True)


def haversine_distance(df):
    df[['Latitude', 'Longitude']] = np.radians(df[['Latitude', 'Longitude']])

    grouped_df = df.groupby('ECTRL ID')[['Latitude', 'Longitude']]
    dlat, dlon = grouped_df.diff().T.values

    # haversine distance
    a = np.sin(dlat / 2.0) ** 2 + np.cos(df['Latitude'].shift(-1)) * np.cos(df['Latitude']) * np.sin(dlon / 2.0) ** 2
    distances = np.r_[np.nan, 6371 * 1000 * 2 * np.arcsin(np.sqrt(a))]

    return distances


flight_points['Time Over'] = pd.to_datetime(flight_points['Time Over'], format='%d-%m-%Y %H:%M:%S')

flight_points['time_delta(s)'] = flight_points.groupby('ECTRL ID')['Time Over'].diff().dt.total_seconds()

flight_points['dist(m)'] = pd.Series(haversine_distance((flight_points)))

flight_points['speed(m/s)'] = flight_points['dist(m)'] / flight_points['time_delta(s)'].shift(1)

flight_points['altitude_delta'] = flight_points.groupby('ECTRL ID')[['Flight Level']].diff()

flight_points['flight_angle(rad)'] = np.arctan(np.divide(flight_points['altitude_delta'], flight_points['dist(m)'],
                                                         out=np.zeros_like(flight_points['altitude_delta']),
                                                         where=flight_points['dist(m)'] != 0))

final_df = flight_points[
    ['ECTRL ID', 'speed(m/s)', 'dist(m)', 'Flight Level', 'Time Over', 'time_delta(s)', 'flight_angle(rad)']].rename(
    columns={'Flight Level': 'altitude(m)', 'Time Over': 'timestamp'})

final_df.to_csv('flight_point_data_.csv', sep=',', index=False, encoding='utf-8') # Change the month and year in the file name
