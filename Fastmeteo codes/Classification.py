#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openap import prop


#%%

# Read csv file
df = pd.read_csv("General data\Flights_20190301_20190331.csv") # Add Flights file for the month and year of the data
EU_airports = pd.read_csv("General data\eu-airports.csv") # Add EU Airports file

#%%


# Rename columns
df.rename(columns={'Actual Distance Flown (nm)':"Distance Flown (km)"}, inplace=True)
df.rename(columns={'STATFOR Market Segment':"TYPE"}, inplace=True)
EU_airports.rename(columns={'ident':"Airport"}, inplace=True)
EU_airports.rename(columns={'country_name':"EU Country"}, inplace=True)


# Converting nautical miles to kilometers 
df['Distance Flown (km)'] = df['Distance Flown (km)'] * 1.852


# Filter data for each segment
df_traditional = df[df['TYPE'] == 'Traditional Scheduled']
df_lowcost = df[df['TYPE'] == 'lowcost']


# Concatenate filtered dataframes into a new dataframe
df1 = pd.concat([df_traditional, df_lowcost], ignore_index=True)


# Limiting to only European airports
df2 = df1[((df1['ADEP'].isin(EU_airports['Airport'])) | (df1['ADES'].isin(EU_airports['Airport']))) & (df1['ADEP'] != df1['ADES'])]


# Limiting aircraft to be in OpenAP
available_aircraft = prop.available_aircraft(use_synonym=True)
filter = [x.upper() for x in available_aircraft]
df3 = df2[df2['AC Type'].str.upper().isin(filter)]

# Merge with EU_airports to get EU Country for ADEP
df3 = pd.merge(df3, EU_airports, how='left', left_on='ADEP', right_on='Airport')


# Merge with EU_airports to get EU Country for ADES
df3 = pd.merge(df3, EU_airports, how='left', left_on='ADES', right_on='Airport', suffixes=('_ADEP', '_ADES'))


# Sort data based on classifications
Short = df3[(df3['Distance Flown (km)'] > 0) & (df3['Distance Flown (km)'] <= 500)]

# Organizing data in an ascending order
Short.sort_values(by='Distance Flown (km)', ascending=True)

# Find ids of flights in short
short_flights_IDS = Short['ECTRL ID']
# %%
