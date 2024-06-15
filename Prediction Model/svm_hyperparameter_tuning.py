import numpy as np
import pandas as pd
from load_data import load_dataset
from main_preprocessing import dataset_split, stanY, inverse_scale
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import expon, uniform
# Load dataset
X, Y = load_dataset()

# Preprocessing
X_numerical = (X - X.min()).dt.days.values
X_train, Y_train, X_test, Y_test = dataset_split(X_numerical, Y, 8/10)
Y_train, Y_test, Y_mean, Y_std = stanY(Y_train, Y_test)

#choose between random search and grid search

def train_binary_svm_regressor(X_train, y_train, params):
    clf = svm.SVR()
    grid_search = GridSearchCV(clf, params, cv=3, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    return best_clf, grid_search.best_params_

# Define the parameters grid
param = {
    'C': [0.001,0.0001,0.1, 1, 0.01,10],
    'gamma': [100,10,1,0.1, 0.01,0.001],
    'kernel': ['linear']
    
}


# def train_binary_svm_regressor(X_train, y_train, params, n_iter):
#     clf = svm.SVR()
#     random_search = RandomizedSearchCV(clf, params, n_iter=n_iter, cv=2, scoring='neg_mean_squared_error', verbose=2, random_state=42)
#     random_search.fit(X_train, y_train)
#     best_clf = random_search.best_estimator_
#     return best_clf, random_search.best_params_

# # Define the parameter distribution
# param = {
#     'C': uniform(0.0001, 10),      # Uniform distribution from 0.001 to 100
#     'gamma': uniform(0.001, 100),  # Uniform distribution from 0.0001 to 10
#     'kernel': ['linear'] 
# }

# # Number of parameter settings that are sampled
# n_iter_search = 20


# Train SVM regressor with hyperparameter tuning
#grid search
best_clf, best_params = train_binary_svm_regressor(X_train.reshape(-1,1), Y_train, param)
#
#random search
#best_clf, best_params = train_binary_svm_regressor(X_train.reshape(-1,1), Y_train, param, n_iter_search)

# Predict using the best model
Y_pred = best_clf.predict(X_test.reshape(-1,1))
y_pred = inverse_scale(Y_pred, Y_mean, Y_std)

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

# Print best model and hyperparameters
print("Best model:")
print(best_clf)
print("\nBest hyperparameters:")
print("Kernel:", best_params['kernel'])
print("C:", best_params['C'])
print("Gamma:", best_params['gamma'])

# Return the best model
best_model = best_clf
