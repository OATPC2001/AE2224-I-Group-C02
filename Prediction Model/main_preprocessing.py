import numpy as np
import pandas as pd
from load_data import load_dataset
from sklearn.model_selection import train_test_split


X, Y = load_dataset()

#X_features = pd.DataFrame()
#X_features['dayofweek'] = X.dt.dayofweek
#0 is Monday
X = X.dt.date

def dataset_split(X, Y, fraction):
    # split data into two subsets {a} and {b} so that:
    #   - {X_train} and {y_train} contain a fraction of the dataset equal to {fraction}
    #   - {X_test} and {y_test} contain a fraction of the dataset equal to {1-fraction}
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0-fraction, random_state=0, shuffle=True)

    # return split datasets
    return X_train, Y_train, X_test, Y_test

def scale(X, a, b):
    # return the scaled data
    return (X - a) / b


def inverse_scale(X, a, b):
    # return the scaled data
    return b * X + a

def standardize(X_train, Y_train, X_test, Y_test):

    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    Y_mean = np.mean(Y_train)
    Y_std = np.std(Y_train)

    # standardize all data
    X_train = scale(X_train,X_mean,X_std)
    Y_train = scale(Y_train, Y_mean,Y_std)
    X_test = scale(X_test,X_mean,X_std)
    Y_test = scale(Y_test,Y_mean,Y_std)
    # return the standardized data and the mean and standard deviation of the training data
    return X_train, Y_train, X_test, Y_test, X_mean, Y_mean, X_std, Y_std



def stanY(Y_train, Y_test):

    Y_mean = np.mean(Y_train)
    Y_std = np.std(Y_train)

    # standardize all data
    Y_train = scale(Y_train, Y_mean, Y_std)
    Y_test = scale(Y_test, Y_mean, Y_std)
    # return the standardized data and the mean and standard deviation of the training data
    return Y_train, Y_test, Y_mean, Y_std