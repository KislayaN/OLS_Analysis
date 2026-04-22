import numpy as np

class BLUE:
    def __init__(self, X=None, y=None):
        super().__init__()
        self.features = X
        self.target = y
        
    def coefficients(self):
        if self.features is None:
            return "features (X) missing for calculation"
        
        if self.target is None:
            return "target (y) missing for calculation"
        
        return np.linalg.inv(np.transpose(self.features) @ self.features) @ np.transpose(self.features) @ self.target