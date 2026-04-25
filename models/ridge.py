import numpy as np

class Ridge:
    def __init__(self, fit_intercept=True, alpha=0.01):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
        
    def add_intercept(self, X):
        return np.column_stack((np.ones(len(X)), X))
    
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_model = self.add_intercept(X_val) if self.fit_intercept else X_val
        
        # Creating identity matrix
        p = X.shape[1]
        Identity = np.eye(p)
        
        # making the first element of diagonal 0 because it would end up
        # penalizing the intercept that we dont want to happen
        Identity = Identity[0, 0] = 0
        
        betas = np.linalg.pinv(X_model.T @ X_model + self.alpha * Identity) @ X_model.T @ y_val
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