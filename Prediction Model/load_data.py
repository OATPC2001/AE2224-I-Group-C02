import pandas as pd
    
def load_dataset():
    
    data = pd.read_csv("total_emissions.csv")
    X = pd.to_datetime(data['Actual Off Block Date'], format='%d-%m-%Y')
    Y = data['CO2']
    return [X,Y]

