import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files
scenario1_df = pd.read_csv("CO2_final_prediction.csv")
scenario2_df = pd.read_csv("CO2_long_haul_svm.csv")
scenario3_df = pd.read_csv("2%_Decrease_CO2.csv")

# Assuming each CSV file has columns 'date' and 'emissions'
# Convert 'date' column to datetime
scenario1_df['Actual Off Block Date'] = pd.to_datetime(scenario1_df['Actual Off Block Date'])
scenario2_df['Actual Off Block Date'] = pd.to_datetime(scenario2_df['Actual Off Block Date'])
scenario3_df['Actual Off Block Date'] = pd.to_datetime(scenario3_df['Actual Off Block Date'])

# Aggregate emissions data over 30 days
scenario1_agg = scenario1_df.groupby(pd.Grouper(key='Actual Off Block Date', freq='30D')).sum()
scenario2_agg = scenario2_df.groupby(pd.Grouper(key='Actual Off Block Date', freq='30D')).sum()
scenario3_agg = scenario3_df.groupby(pd.Grouper(key='Actual Off Block Date', freq='30D')).sum()


scenario1_agg['CO2'] = scenario1_agg['CO2'] / 1e6
scenario2_agg['CO2'] = scenario2_agg['CO2'] / 1e6
scenario3_agg['CO2'] = scenario3_agg['CO2'] / 1e6

# Plot aggregated data
plt.figure(figsize=(10, 6))
plt.plot(scenario1_agg.index, scenario1_agg['CO2'], label='Scenario 1')
plt.plot(scenario2_agg.index, scenario2_agg['CO2'], label='Scenario 2')
plt.plot(scenario3_agg.index, scenario3_agg['CO2'], label='Scenario 3')

#plt.title('Monthly Emissions')
plt.xlabel('Year')
plt.ylabel('Monthly Total CO2 Emissions (in thousands of tonnes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('combined.png')
plt.show()