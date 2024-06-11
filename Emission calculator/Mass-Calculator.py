#%%
import numpy as np
import pandas as pd
import openap
from openap import Emission
from openap import prop

#%%

actype = prop.available_aircraft(use_synonym = True)
actype = [x.upper() for x in actype]

df_flights = pd.read_csv('Emission calculator\flight_point_data_.csv') # Add Flight Points Data file for the month and year of the data

types = pd.read_csv('Data\Flights_20190301_20190331.csv.gz')[['ECTRL ID', 'AC Type']] # Add Flights file for the month and year of the data
types = types[types['AC Type'].isin(actype)]
take_off_weights = pd.read_csv('Data\mtow.csv').rename(columns={'Name': 'AC Type'})
take_off_weights['AC Type'] = take_off_weights['AC Type'].str.upper()
types = pd.merge(left=types, right=take_off_weights, on='AC Type', how='inner')

df_flights = df_flights.merge(right=types[['ECTRL ID', 'MTOW', 'AC Type']], how='left', on='ECTRL ID').dropna()

identities = df_flights['ECTRL ID'].unique().tolist()
print(f'Total number of flights: {len(identities)}')

count = 0

for number in identities:

    if count % 5000 == 0:
        print(f'{count} counts completed.')
    count += 1

    aircraft = df_flights[df_flights['ECTRL ID'] == number]['AC Type'].values[0].upper()
 
    df_identity = df_flights[df_flights['ECTRL ID'] == number]

    ff = openap.FuelFlow(aircraft, use_synonym=True)

    time_stamp = df_identity['timestamp']
    path_angles = df_identity['flight_angle(rad)']*(180/np.pi)
    time_deltas = df_identity['time_delta(s)']
    tasa = df_identity['speed(m/s)']*1.94384
    alta = df_identity['altitude(m)']*3.28084
    acdict = prop.aircraft(aircraft)
    MTOW = acdict["limits"]["MTOW"]
    
    max_passengers = acdict["pax"]["max"]
    W_const = acdict["limits"]["OEW"]+0.8*max_passengers*84
    mass0 = MTOW
    for i in range(3):
        mass=mass0
        n = 0
        for (dt, tas, alt, pa) in zip(time_deltas, tasa, alta, path_angles):
            FF = ff.enroute(mass, tas, alt, pa)
            n += 1
            if n == 6:
                FF_cruise = FF
            mass -= FF * dt

        fuel = mass0 - mass
        mass0 = W_const + fuel + FF_cruise*30*60

        if mass0 > MTOW:
            mass0 = MTOW

    df_flights.loc[df_flights['ECTRL ID'] == number, 'Fuel'] = fuel
    df_flights.loc[df_flights['ECTRL ID'] == number, 'Total Mass'] = mass0

df_flights.to_csv('masses19.csv', sep=',', index=False, encoding='utf-8') # Change the month and year in the file name
# %%