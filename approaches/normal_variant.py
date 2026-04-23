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
        
    def _add_intercept(self, X):
        self.features = np.column_stack((np.ones(len(X)), X))
        
    def fit(self, X, target):
        X_model = self._add_intercept(X) if self.fit_intercept else X
        
        # Using pinv (pseudo-inverse) is safer than inv for singular matrices
        self.coefficients = np.linalg.pinv(X_model.T @ X_model) @ X_model.T @ target
        
        if self.fit_intercept: 
            self.intercept = self.coefficients[0]
            self.coefficients = self.coefficients[1:]
        else: 
            self.intercept = 0.0
    
    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call .fit() first")
        
        X_model = self._add_intercept if self.fit_intercept else X
        
        self.predicted_value = X_model @ self.coefficients 
        return self.predicted_value
    
    def score(self, X, y):
        y_pred = self.predict(X)
        r2_score = self.get_metric.R2_Score(y=y, y_pred=y_pred)
        
        return r2_score
    
    def summary(self, X, y):
        summary_dict = {}
        
        y_pred = self.predict(X)
        
        summary_dict['Coefficients'] = self.coefficients[1:] if self.fit_intercept else self.coefficients
        summary_dict['Intercept'] = self.intercept
        summary_dict['MSE'] = self.get_metric.MSE_score(observed_value=y, predicted_value=y_pred)
        summary_dict['R2'] = self.get_metric.R2_Score(y=y, y_pred=y_pred)
        
        return summary_dict
        
    def resid(self, y, y_pred):
        self.residuals = y - y_pred
        return self.residuals