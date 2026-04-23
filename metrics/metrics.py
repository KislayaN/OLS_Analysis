import numpy as np

class Metric:
    def __init__(self):
        self.MSE = None
        self.R2 = None
    
    def MSE_score(self, observed_value, predicted_value): 
        n = len(observed_value)
        
        error = observed_value - predicted_value 
        mse = np.sum(error ** 2, axis=0) / n
        
        self.MSE = mse
        return mse
    
    def R2_Score(self, y, y_pred):
        y_mean = y.mean()
        y_ = y - y_mean
        
        # Total sum of Squares
        SS_tot = np.sum(y_, axis=0)
        
        # Residual sum of Squares
        SS_res = np.sum((y - y_pred) ** 2, axis=0)
        
        r2_score = 1 - (SS_res / SS_tot)
        self.R2 = r2_score
        return r2_score
    