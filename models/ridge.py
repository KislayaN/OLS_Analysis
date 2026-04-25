import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from metrics.metrics import Metric

import numpy as np

class Ridge:
    def __init__(self, fit_intercept=True, alpha=0.01):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
        self.final_mse = 0.0
        self.get_metric = Metric()
        
    def add_intercept(self, X):
        return np.column_stack((np.ones(len(X)), X))
    
    def calculate_mse(self, prediction, target):
        self.final_mse = self.get_metric.MSE_score(
            predicted_value=prediction,
            observed_value=target
        )
    
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_model = self.add_intercept(X_val) if self.fit_intercept else X_val
        
        # Creating identity matrix
        p = X_model.shape[1]
        Identity = np.eye(p)
        
        # making the first element of diagonal 0 because it would end up
        # penalizing the intercept that we dont want to happen
        Identity[0, 0] = 0
        
        betas = np.linalg.pinv((X_model.T @ X_model) + (self.alpha * Identity)) @ (X_model.T @ y_val)
        betas = np.array(betas).flatten()
        
        if self.fit_intercept:
            self.coefficients = betas[1: ]
            self.intercept = betas[0]
        else: 
            self.coefficients = betas
            self.intercept = 0.0
            
    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call .fit() first")
        
        X_val = X.values if hasattr(X, 'values') else X
        X_model = self.add_intercept(X) if self.fit_intercept else X_val
        
        if self.fit_intercept:
            beta_full = np.insert(self.coefficients, 0, self.intercept)
            return X_model @ beta_full
        
        return X_model @ self.coefficients