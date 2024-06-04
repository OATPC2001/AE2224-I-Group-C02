import numpy as np
import pandas as pd
import openap

pd.set_option('display.max_columns', None)

df_flights = pd.read_csv('Emission calculator\masses19.csv')
take_off_weights = pd.read_csv('Emission calculator\mtow.csv').rename(columns={'Name': 'AC Type'})
types = pd.read_csv('General data\Flights_20190301_20190331.csv')[['ECTRL ID', 'AC Type']] # Add Flights file for the month and year of the data

take_off_weights['AC Type'] = take_off_weights['AC Type'].str.upper()


inter_df = df_flights[['AC Type']]


take_off_weights['emission'] = take_off_weights['AC Type'].apply(lambda x: openap.Emission(x))
take_off_weights['fuel_flow'] = take_off_weights['AC Type'].apply(lambda x: openap.FuelFlow(x, use_synonym=True))

types = pd.merge(left=types, right=take_off_weights, on='AC Type', how='inner')

df_flights = df_flights.merge(right=types, how='left', on='ECTRL ID').dropna()



df_flights['flight_angle(deg)'] = (df_flights['flight_angle(rad)'] * 360) / (2 * np.pi)
df_flights['speed(kt)'] = df_flights['speed(m/s)'] * 1.9438452
df_flights['altitude(ft)'] = df_flights['altitude(m)'] * 3.28084




overspeed_flights = df_flights[df_flights['speed(m/s)'] > 400]
grouped_over = overspeed_flights.groupby('ECTRL ID')
grouped_over_speed = grouped_over.filter(lambda x: (x['speed(m/s)'] > 600).any())
grouped_over_length = grouped_over.filter(lambda x: len(x) > 10)
grouped_over_time = grouped_over.filter(lambda x: x['time_delta(s)'].max() < 60)

unique_ids = pd.concat(
    [grouped_over_speed['ECTRL ID'], grouped_over_time['ECTRL ID'], grouped_over_length['ECTRL ID']]).unique().tolist()

df_flights = df_flights[~df_flights['ECTRL ID'].isin(unique_ids)]


grouped_df_flights = df_flights.groupby('ECTRL ID')

final_df = pd.DataFrame(columns=['ECTRL ID', 'CO2'])


# Initialize variables for emissions
mass = 0
CO2 = 0
H2O = 0
NOx = 0
CO = 0
HC = 0
count = 0

for group_key, group_df in grouped_df_flights:
    count += 1

    mass = group_df['Total Mass'].iloc[0]  # Initial mass

    CO2 = 0  # g/s
    H2O = 0  # g/s
    NOx = 0  # g/s
    CO = 0  # g/s
    HC = 0  # g/s
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

    if count % 5000 == 0:
        print(count)

final_df.to_csv('emissions.csv', sep=',', index=False, encoding='utf-8') # Change the name of the file to the month and year of the data

print('Done')