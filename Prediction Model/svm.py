import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_dataset
from main_preprocessing import dataset_split, stanY, inverse_scale
from plot import plot_scatter,plot_dual_scatter,plot_line
from showdate import date_to_numeric, numeric_to_date
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, Y = load_dataset()

X_numerical = (X-X.min()).dt.days.values
X_train, Y_train, X_test, Y_test = dataset_split(X_numerical, Y, 8/10)
Y_train, Y_test, Y_mean, Y_std = stanY(Y_train, Y_test)



def train_binary_svm_regressor(X_train, y_train, C, gamma):
    clf = svm.SVR(C=C, kernel='linear', gamma=gamma)
    clf.fit(X_train, y_train)
    return clf

#kernel rbf decent but cannot predict,linear can predict but takes ages cause too many data 

clf = train_binary_svm_regressor(X_train.reshape(-1,1),Y_train,0.1,10)

Y_pred = clf.predict(X_test.reshape(-1,1))
y_pred = inverse_scale(Y_pred, Y_mean, Y_std)

#0.1,10

#Y_pred used for checking errors so its fine both are scaled
# Mean Absolute Error (MAE)

mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error:", mae)

# Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-squared (R2)
r2 = r2_score(Y_test, Y_pred)
print("R-squared (R2):", r2)


X_test = X_test.tolist()

lst= []
for element in X_test:
    a = numeric_to_date(element)
    lst.append(a)

X_test = np.array(lst)


df = pd.DataFrame({'Actual Off Block Date': X_test ,'Emissions': y_pred})
df['Actual Off Block Date'] = pd.to_datetime(df['Actual Off Block Date'], format='%d-%m-%Y')
df_sorted = df.sort_values(by='Actual Off Block Date',ascending = True)
df_sorted['Actual Off Block Date'] = pd.to_datetime(df['Actual Off Block Date'], format='%Y-%m-%d')
df_sorted['Actual Off Block Date'] = df_sorted['Actual Off Block Date'].dt.strftime('%d-%m-%Y')
df_test = df_sorted
X_test = pd.to_datetime(df_sorted['Actual Off Block Date'], format='%d-%m-%Y')
y_pred = df_test['Emissions']

date_end = '15-03-2035'
n_dates = date_to_numeric(date_end)
X_future = np.arange(0, n_dates +1 )

predicted = clf.predict(np.array(X_future).reshape(-1,1))
predict = inverse_scale(predicted, Y_mean, Y_std)

X_future = X_future.tolist()

lsa= []
for element in X_future:
    a = numeric_to_date(element)
    lsa.append(a)

X_future = np.array(lsa)

dff = pd.DataFrame({'Actual Off Block Date': X_future ,'Emissions': predict})
dff['Actual Off Block Date'] = pd.to_datetime(dff['Actual Off Block Date'], format='%d-%m-%Y')
dff_sorted = dff.sort_values(by='Actual Off Block Date',ascending = True)
dff_sorted['Actual Off Block Date'] = pd.to_datetime(dff['Actual Off Block Date'], format='%Y-%m-%d')
dff_sorted['Actual Off Block Date'] = dff_sorted['Actual Off Block Date'].dt.strftime('%d-%m-%Y')

X_future = pd.to_datetime(dff_sorted['Actual Off Block Date'], format='%d-%m-%Y')
predict = dff_sorted['Emissions'] 
dff_sorted.to_csv("CO2_long_haul_svm.csv",index=False)

#plot_line(X_future,predict,'CO2_svm_future_long_tuned.png','blue')

