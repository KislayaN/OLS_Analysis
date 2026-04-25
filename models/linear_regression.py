from sklearn.linear_model import LinearRegression

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from metrics.metrics import Metric

class Linear_Regression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.lr = LinearRegression()
        self.get_metrics = Metric()
        self.final_mse = 0.0
    
    def fit(self, X, y):
        if self.fit_intercept:
            self.lr.fit_intercept = self.fit_intercept
        
        return self.lr.fit(X, y)
    
    def predict(self, X):
        return self.lr.predict(X)
        
    def calculate_mse(self, prediction, target):
        self.final_mse =  self.get_metrics.MSE_score(
            observed_value=target,
            predicted_value=prediction
        )