import numpy as np
import pandas as pd
import openap

pd.set_option('display.max_columns', None)

# Preprocessing

# Read flight data
df_flights = pd.read_csv('Masses2019.csv')  # Put the location of your file here !!!!

# Read take-off weights data
take_off_weights = pd.read_csv('Emission calculator\mtow.csv').rename(columns={'Name': 'AC Type'})

# Read aircraft types data
types = pd.read_csv('Python files\Flights_20190301_20190331.csv.gz')[['ECTRL ID', 'AC Type']]

# Convert aircraft type to uppercase
take_off_weights['AC Type'] = take_off_weights['AC Type'].str.upper()

print('Check 1')
print(len(df_flights))
print(len(df_flights.groupby('ECTRL ID')))

# Calculate emissions and fuel flow for each aircraft type
take_off_weights['emission'] = take_off_weights['AC Type'].apply(lambda x: openap.Emission(x))
take_off_weights['fuel_flow'] = take_off_weights['AC Type'].apply(lambda x: openap.FuelFlow(x, use_synonym=True))

# Merge aircraft types with flight data
types = pd.merge(left=types, right=take_off_weights, on='AC Type', how='inner')
df_flights = df_flights.merge(right=types, how='left', on='ECTRL ID').dropna()

# Convert units
df_flights['flight_angle(deg)'] = (df_flights['flight_angle(rad)'] * 360) / (2 * np.pi)
df_flights['speed(kt)'] = df_flights['speed(m/s)'] * 1.9438452
df_flights['altitude(ft)'] = df_flights['altitude(m)'] * 3.28084

print('Check 2')
print(len(df_flights))
print(len(df_flights.groupby('ECTRL ID')))

# Filter overspeed flights
overspeed_flights = df_flights[df_flights['speed(m/s)'] > 500]
grouped_over = overspeed_flights.groupby('ECTRL ID')

# Filter overspeed flights with speed > 600
grouped_over_speed = grouped_over.filter(lambda x: (x['speed(m/s)'] > 600).any())

# Filter overspeed flights with length > 10
grouped_over_length = grouped_over.filter(lambda x: len(x) > 10)

# Filter overspeed flights with max time delta < 60
grouped_over_time = grouped_over.filter(lambda x: x['time_delta(s)'].max() < 60)

# Get unique IDs of filtered flights
unique_ids = pd.concat(
    [grouped_over_speed['ECTRL ID'], grouped_over_time['ECTRL ID'], grouped_over_length['ECTRL ID']]).unique().tolist()

# Remove flights with unique IDs
df_flights = df_flights[~df_flights['ECTRL ID'].isin(unique_ids)]

print('Check 3')

grouped_df_flights = df_flights.groupby('ECTRL ID')

print(len(df_flights))
print(len(grouped_df_flights))

final_df = pd.DataFrame(columns=['ECTRL ID', 'CO2'])

# Initialize variables for emissions
mass = 0
count = 0

# Calculate emissions for each flight
for group_key, group_df in grouped_df_flights:
    count += 1
    # Reset values when encountering a new ECTRL ID
    mass = group_df['Total Mass'].iloc[0]  # Initial mass
    CO2 = 0  # g/s
    ff = group_df['fuel_flow'].iloc[0]  # FuelFlow object
    emission = group_df['emission'].iloc[0]  # Emission object

    # Loop over time deltas
    for (dt, tas, alt, pa) in zip(group_df['time_delta(s)'].values, group_df['speed(kt)'].values,
                                  group_df['altitude(ft)'].values,
                                  group_df['flight_angle(deg)'].values):
        try:
            FF = ff.enroute(mass, tas, alt, pa)
        except Exception as e:
            print("Values causing the Error with group key: ", group_key)
            print('tas: ', tas)
            print('values: ', group_df['speed(kt)'].values)
            exit()

        mass -= FF * dt
        try:
            CO2 += emission.co2(FF) * dt / 1000
        except Exception as e:
            print("Values causing the Error with group key: ", group_key)
            print('tas: ', tas)
            print('values: ', group_df['speed(kt)'].values)
            exit()

    entry = [group_key, CO2]
    final_df.loc[len(final_df)] = entry

    if count % 50000 == 0:
        print(f'{count} of {len(df_flights)} counts completed ({count / len(df_flights) * 100}%)')

# Save final results to a CSV file
final_df.to_csv('Emissions2019.csv', sep=',', index=False, encoding='utf-8')  # Rename it to the correct name

print('Done')