import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from metrics.metrics import Metric

class OLS:
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.get_metric = Metric()
        self.coefficients = None
        self.intercept = None
        self.final_mse = 0.0
        
    def _add_intercept(self, X):
        return np.column_stack((np.ones(len(X)), X))
    
    def calculate_mse(self, prediction, target):
        self.final_mse = self.get_metric.MSE_score(
            predicted_value=prediction,
            observed_value=target
        )
        return self.final_mse
        
    def fit(self, X, target):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = target.values if hasattr(target, 'values') else target
        
        X_model = self._add_intercept(X_val) if self.fit_intercept else X_val
        
        # Using pinv (pseudo-inverse) is safer than inv for singular matrices
        beta_full = np.linalg.pinv(X_model.T @ X_model) @ X_model.T @ y_val
        beta_full = np.array(beta_full).flatten()
        
        if self.fit_intercept: 
            self.intercept = beta_full[0]
            self.coefficients = beta_full[1: ]
        else: 
            self.intercept = 0.0
            self.coefficients = beta_full
    
    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call .fit() first")
        
        X_val = X.values if hasattr(X, 'values') else X
        X_model = self._add_intercept(X_val) if self.fit_intercept else X_val
        
        if self.fit_intercept:
            beta_full = np.insert(self.coefficients, 0, self.intercept)
            return X_model @ beta_full
        
        return X_model @ self.coefficients
    
    def score(self, X, y):
        y_pred = self.predict(X)
        r2_score = self.get_metric.R2_Score(y=y, y_pred=y_pred)
        
        return r2_score
    
    def summary(self, X, y):
        summary_dict = {}
        
        y_pred = self.predict(X)
        
        summary_dict['Coefficients'] = self.coefficients[1: ] if self.fit_intercept else self.coefficients
        summary_dict['Intercept'] = self.intercept
        summary_dict['MSE'] = self.get_metric.MSE_score(observed_value=y, predicted_value=y_pred)
        summary_dict['R2'] = self.get_metric.R2_Score(y=y, y_pred=y_pred)
        
        return summary_dict