import numpy as np

class OLS:
    def __init__(self):
        super().__init__()
        self.features = None
        self.target = None
        self.coefficients = None
        
    def fit(self, X, target):
        self.features = X
        self.target = target
        
        if self.features is None:
            return "features (X) missing for calculation"
        
        if self.target is None:
            return "target (y) missing for calculation"
        
        self.coefficients = np.linalg.inv(np.transpose(self.features) @ self.features) @ np.transpose(self.features) @ self.target
        return self.coefficients
    
    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Coefficients are not estimated, run .coefficients()")
        
        prediction = X @ self.coefficients 
        return prediction