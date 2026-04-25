import numpy as np

class Lasso:
    def __init__(self, fit_intercept=True, iterations=1000, learning_rate=0.01, tol=1e-6, alpha=0.001):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.n_iter = iterations
        self.tol = tol
        self.cost_history = []
        self.weights = None
        self.alpha = alpha
        
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        if self.fit_intercept:
            X_model = np.column_stack((np.ones(len(X_val)), X_val))
        
        m, n = X_model.shape
        
        self.weights = np.zeros(n)
        y_model = np.asarray(y_val).reshape(-1)
        
        for epoch in range(self.n_iter):
            
            prediction = X_model @ self.weights
            error = prediction - y_model
            
            cost = (1/m) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            if self.fit_intercept:
                l1_penalty = np.sign(self.weights)
                l1_penalty[0, :] = 0
            
            del_J = (2/m) * (X_model.T @ error) + (self.alpha * l1_penalty)
            
            if np.linalg.norm(del_J) < self.tol:
                break
            
            self.weights -= self.learning_rate * del_J
        
        return self.weights
    
    def predict(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        
        if self.fit_intercept:
            X_model = np.column_stack((np.ones(len(X_val)), X_val))
            return X_model @ self.weights
        
        return X_val @ self.weights
            