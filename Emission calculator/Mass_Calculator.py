import numpy as np
import pandas as pd
import openap
from openap import prop

pd.set_option('display.max_columns', 6)

# Read csv files
flights = pd.read_csv('Python files\Flights_20190301_20190331.csv.gz')  # change file path
types = flights[['ECTRL ID', 'AC Type']]
df_flights = pd.read_csv('Flight_points_with_speed_2019.csv')  # change file path
EU_airports = pd.read_csv("General data\eu-airports.csv")  # change file path

# Rename columns
flights.rename(columns={'Actual Distance Flown (nm)': "Distance Flown (km)", 'STATFOR Market Segment': "TYPE"},
               inplace=True)
EU_airports.rename(columns={'ident': "Airport", 'country_name': "EU Country"}, inplace=True)

# Convert nautical miles to kilometers
flights['Distance Flown (km)'] = flights['Distance Flown (km)'] * 1.852

# Filter data for each segment
df_traditional = flights[flights['TYPE'] == 'Traditional Scheduled']
df_lowcost = flights[flights['TYPE'] == 'Low-cost']

# Combine and filter by ECTRL ID
combined_flights = pd.concat([df_traditional, df_lowcost])
types_filter = combined_flights['ECTRL ID'].unique().tolist()
flights = flights[flights['ECTRL ID'].isin(types_filter)]
df_flights = df_flights[df_flights['ECTRL ID'].isin(types_filter)]

print(len(df_flights.groupby('ECTRL ID')))
print(len(df_flights))

print('speed filtering : ---------------------------------------')
# print(flights[0:50])

# Filter for overspeed flights
overspeed_flights = df_flights[df_flights['speed(m/s)'] > 500]
grouped_over = overspeed_flights.groupby('ECTRL ID')
grouped_over_speed = grouped_over.filter(lambda x: (x['speed(m/s)'] > 600).any())
grouped_over_length = grouped_over.filter(lambda x: len(x) > 10)
grouped_over_time = grouped_over.filter(lambda x: x['time_delta(s)'].max() < 60)

# Combine filters
speeds_filter = pd.concat([
    grouped_over_time['ECTRL ID'],
    grouped_over_length['ECTRL ID']
]).unique()

# Further filtering
df_flights = df_flights.drop(grouped_over_speed.index)
df_flights = df_flights[~df_flights['ECTRL ID'].isin(speeds_filter)]
flights = flights[flights['ECTRL ID'].isin(df_flights['ECTRL ID'].unique().tolist())]

print(len(df_flights.groupby('ECTRL ID')))
print(len(df_flights))

print('airport filtering --------------------------------------------------------------------')

# Map country names to departure and arrival airports
flights['Departure Country'] = flights['ADEP'].map(EU_airports.set_index('Airport')['EU Country'])
flights['Arrival Country'] = flights['ADES'].map(EU_airports.set_index('Airport')['EU Country'])
flights = flights.dropna(subset=['ADEP', 'ADES'], how='all')
df_flights = df_flights[df_flights['ECTRL ID'].isin(flights['ECTRL ID'].unique().tolist())]

print(len(df_flights.groupby('ECTRL ID')))
print(len(df_flights))

# Filter aircraft types
actype = prop.available_aircraft(use_synonym=True)
actype = [x.upper() for x in actype]
types = types[types['AC Type'].isin(actype)]

print('AC type filtering ----------------------------------------------------------------')

# Merge with take-off weights
take_off_weights = pd.read_csv('Emission calculator\mtow.csv').rename(columns={'Name': 'AC Type'})
take_off_weights['AC Type'] = take_off_weights['AC Type'].str.upper()
types = pd.merge(left=types, right=take_off_weights, on='AC Type', how='inner')

# Merge with df_flights
df_flights = df_flights.merge(right=types[['ECTRL ID', 'MTOW', 'AC Type']], how='left', on='ECTRL ID').dropna()

identities = df_flights['ECTRL ID'].unique().tolist()
print(len(identities))

count = 0

for number in identities:
    if count % 5000 == 0:
        print(f'{count} (of {len(identities)}) counts completed.')
    count += 1

    aircraft = df_flights[df_flights['ECTRL ID'] == number]['AC Type'].values[0].upper()
    df_identity = df_flights[df_flights['ECTRL ID'] == number]

    ff = openap.FuelFlow(aircraft, use_synonym=True)
    time_stamp = df_identity['timestamp']
    path_angles = df_identity['flight_angle(rad)'] * (180 / np.pi)
    time_deltas = df_identity['time_delta(s)']
    tasa = df_identity['speed(m/s)'] * 1.94384
    alta = df_identity['altitude(m)'] * 3.28084
    acdict = prop.aircraft(aircraft)
    MTOW = acdict["limits"]["MTOW"]

    max_passengers = acdict["pax"]["max"]
    W_const = acdict["limits"]["OEW"] + 0.8 * max_passengers * 84
    mass0 = MTOW

    for i in range(3):
        mass = mass0
        n = 0
        for (dt, tas, alt, pa) in zip(time_deltas, tasa, alta, path_angles):
            FF = ff.enroute(mass, tas, alt, pa)
            n += 1
            if n == 6:
                FF_cruise = FF
            mass -= FF * dt

        fuel = mass0 - mass
        mass0 = W_const + fuel + FF_cruise * 30 * 60

        if mass0 > MTOW:
            mass0 = MTOW

    df_flights.loc[df_flights['ECTRL ID'] == number, 'Fuel'] = fuel
    df_flights.loc[df_flights['ECTRL ID'] == number, 'Total Mass'] = mass0

print(df_flights)

df_flights.to_csv('Masses2019.csv', sep=',', index=False, encoding='utf-8')
