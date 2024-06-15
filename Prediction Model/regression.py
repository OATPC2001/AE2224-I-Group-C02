from sklearn.linear_model import LinearRegression
from main_preprocessing import scale, inverse_scale
import numpy as np

def train(X_train,Y_train):
    model = LinearRegression()
    model.fit(X_train,Y_train)
    return model, model.intercept_, model.coef_

def predict(Predictee, model, X_mean, X_std, Y_mean, Y_std):
    Predictee = scale(Predictee, X_mean, X_std)
    Predictee = np.array(Predictee).reshape(-1,1)
    pred = model.predict(Predictee)
    Y_pred = inverse_scale(pred,Y_mean,Y_std)
    return Y_pred


