import numpy as np

class Gradient_Descent_OLS:
    def __init__(self, iterations=1000, learning_rate=0.01, tol=1e-6, fit_intercept=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iter = iterations
        self.tol = tol
        self.cost_history = []
        self.weights = None
        self.intercept = None
        self.fit_intercept = fit_intercept
        
    def add_intercept(self, X):
        return np.column_stack((np.ones(len(X)), X))
        
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_model = self.add_intercept(X_val) if self.fit_intercept else X_val
        m, n = X_model.shape
        
        self.weights = np.zeros(n)
        
        y = np.asarray(y_val).reshape(-1)
        
        for epoch in range(self.n_iter):

            prediction = X_model @ self.weights
            error = prediction - y
            
            cost = (1/m) * np.sum(error**2)
            self.cost_history.append(cost)
            
            # J = (((X_b @ self.weights) - y) ** 2) / m ----- objective function
            
            # del_J = (2/m) * (X_b.T @ ((X_b @ self.weights) - y) )
            del_J = (2/m) * (X_model.T @ error)
            
            if np.linalg.norm(del_J) < self.tol:
                break
            
            self.weights -= self.learning_rate * del_J
        
        if self.fit_intercept:
            self.intercept = self.weights[0]
            self.weights = self.weights[1: ]
        else: 
            self.intercept = 0.0
            self.weights = self.weights
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted yey. Call .fit() first")
        
        X_val = X.values if hasattr(X, 'values') else X
        X_model = self.add_intercept(X_val) if self.fit_intercept else X_val
        
        if self.fit_intercept:
            beta_full = np.insert(self.weights, 0, self.intercept)
            return X_model @ beta_full
    
        return X_model @ self.weights
