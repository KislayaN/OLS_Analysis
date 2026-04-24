from sklearn.preprocessing import StandardScaler
import numpy as np

class Scaler:
    def __init__(self, perform_scaling=False):
        super().__init__()
        self.perform_scaling = perform_scaling
        self.scaler = StandardScaler()
        
    def fit(self, X):
        self.scalable_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[self.scalable_cols])
        
    def transform(self, X):
        if not hasattr(self, "scalable_cols"):
            raise RuntimeError("Scaler must be fitted before calling transform()")

        X = X.copy()
        scaled_values = self.scaler.transform(X[self.scalable_cols])
        X[self.scalable_cols] = scaled_values.astype(float)

        return X