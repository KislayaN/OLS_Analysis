import numpy as np

class BLUE:
    def __init__(self, feature_space=None, target=None):
        super().__init__()
        self.features = feature_space
        self.target = target
        
    def coefficients(self):
        if self.features is None:
            return "features (X) missing for calculation"
        
        if self.target is None:
            return "target (y) missing for calculation"
        
        return np.linalg.inv(np.transpose(self.features) @ self.features) @ np.transpose(self.features) @ self.target