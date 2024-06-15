import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_dataset
from main_preprocessing import dataset_split, stanY, inverse_scale
from plot import plot_scatter, plot_dual_scatter, plot_line
from showdate import date_to_numeric, numeric_to_date
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def add_decrease_factor(predictions, decrease_rate):
    factor = (1 - decrease_rate) ** np.arange(len(predictions))
    return predictions * factor

# Load and preprocess data
deg = 1
X, Y = load_dataset()
X_numerical = (X - X.min()).dt.days.values
X_train, Y_train, X_test, Y_test = dataset_split(X_numerical, Y, 8 / 10)
Y_train, Y_test, Y_mean, Y_std = stanY(Y_train, Y_test)

# Train the polynomial regression model
poly = PolynomialFeatures(degree=deg)
X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly.transform(X_test.reshape(-1, 1))

model = LinearRegression()
model.fit(X_train_poly, Y_train)

Y_pred = model.predict(X_test_poly)
y_pred = inverse_scale(Y_pred, Y_mean, Y_std)

# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error:", mae)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
r2 = r2_score(Y_test, Y_pred)
print("R-squared (R2):", r2)

# Convert numerical dates back to datetime for plotting
X_test = X_test.tolist()
X_test_dates = np.array([numeric_to_date(int(x)) for x in X_test])
X_test_dates = pd.to_datetime(X_test_dates, format='%d-%m-%Y')

# Save the actual vs predicted results
df = pd.DataFrame({'Actual Off Block Date': X_test_dates, 'Number of Flights': y_pred})
df_sorted = df.sort_values(by='Actual Off Block Date', ascending=True)
df_sorted.to_csv('sorted_new.csv', index=False)

# Plot actual vs predicted
# plot_dual_scatter(X, Y, X_test_dates, y_pred, 'actual_vs_test.png')
# plot_line(X_test_dates, y_pred, 'one.png', 'red')

# Predict future values
date_end = '23-03-2035'
n_dates = date_to_numeric(date_end)
X_future = np.arange(0, n_dates + 1)
X_future_poly = poly.transform(X_future.reshape(-1, 1))
predicted = model.predict(X_future_poly)
predict = inverse_scale(predicted, Y_mean, Y_std)
X_future_dates = np.array([numeric_to_date(int(x)) for x in X_future])
X_future_dates = pd.to_datetime(X_future_dates, format='%d-%m-%Y')

# Define scenarios
scenarios = {
    'No Change': predict,
    '2% Decrease': add_decrease_factor(predict, 0.00002)
}

# Plot and save the future predictions for each scenario
for scenario, values in scenarios.items():
    dff = pd.DataFrame({'Actual Off Block Date': X_future_dates, 'CO2': values})
    dff_sorted = dff.sort_values(by='Actual Off Block Date', ascending=True)
    dff_sorted.to_csv(f'{scenario.replace(" ", "_")}_CO2_final_prediction.csv', index=False)
    plot_line(X_future_dates, values, f'{scenario.replace(" ", "_")}_CO2_final_prediction.png', 'red')

#for estimating long haul flight emissions use the cleaned csv file while loading the data and omit the scenario with 2 percent annual decrease

# Final combined plot
plt.figure(figsize=(10, 6))
for scenario, values in scenarios.items():
    plt.plot(X_future_dates, values, label=scenario)
plt.xlabel('Date')
plt.ylabel('CO2')
plt.title('Future Predictions under Different Scenarios')
plt.legend()
#plt.savefig('all_scenarios.png')
plt.show()