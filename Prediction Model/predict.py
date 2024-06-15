import numpy as np
from main_preprocessing import inverse_scale
def predict(z,model,poly,Y_mean,Y_std):
    z = np.array(z)
    z = poly.transform(z.reshape(-1,1))
    z_pred = model.predict(z)
    z_true = inverse_scale(z_pred, Y_mean,Y_std)
    return z_true
