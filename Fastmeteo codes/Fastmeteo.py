#%%

import pandas as pd
from fastmeteo import Grid
import numpy as np
from Classification import short_flights_IDS
import shutil

#%%
# Read csv file
all_data = pd.read_csv("General data\Flight_Points_Actual_20190301_20190331.csv.gz") # Add Flight Points Actual file for the month and year of the data

# Filter flights based on ECTRL IDs
df = all_data[all_data['ECTRL ID'].isin(short_flights_IDS)]

# Data can be limited to a specific range for testing purposes or to break down the data into smaller chunks to save memory
df = df.iloc[0:] 
#%%

timestamp = pd.to_datetime(df['Time Over'], format = '%d-%m-%Y %H:%M:%S')
latitude = df['Latitude'].tolist()
longitude = df['Longitude'].tolist()
altitude = [alt * 100 for alt in df['Flight Level'].tolist()]  # Converting flight level to meters

ECTRL_id = df['ECTRL ID'].tolist()

#%%

flight = pd.DataFrame(
    {
        "timestamp": timestamp,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "ECTRL ID": ECTRL_id,
    }
)

flight = flight.drop_duplicates(["timestamp"])

# Set local_store to the desired directory for temporary files
fmg = Grid(local_store = "/tmp/era5-zarr")

try:
    # Obtain weather information.
    flight_new = fmg.interpolate(flight).copy()

    flight_new = flight_new.dropna()

    # Combine 'u_component_of_wind' and 'v_component_of_wind' into one vector
    wind_vector = np.column_stack((flight_new['u_component_of_wind'], flight_new['v_component_of_wind']))

    # Calculate magnitude and angle of the velocity vector
    magnitude = np.linalg.norm(wind_vector, axis=1)
    angle = np.arctan2(wind_vector[:, 1], wind_vector[:, 0]) * (180 / np.pi)

    # Add magnitude and angle as new columns to the DataFrame
    flight_new.loc[:, 'velocity_magnitude'] = magnitude
    flight_new.loc[:, 'velocity_angle'] = angle

except Exception as e:
    print("An error occurred: ", e)

finally:
    # Save the DataFrame to the specified directory
    flight_new.to_csv('Wind velocities_.csv', sep = ',', index = False, encoding = 'utf-8') # Change the name of the file based on the number of data chunks

    print("Data saved successfully.")

    shutil.rmtree("/tmp/era5-zarr")
    print("Temporary files deleted.")
# %%