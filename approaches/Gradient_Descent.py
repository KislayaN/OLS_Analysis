import numpy as np

class Gradient_Descent_OLS:
    def __init__(self, iterations=1000, learning_rate=0.01, tol=1e-6):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iter = iterations
        self.tol = tol
        self.cost_history = []
        self.weights = None
        
    def fit(self, X, y):
        X_b = np.column_stack((np.ones(len(X)), X))
        m, n = X_b.shape
        
        self.weights = np.zeros(n)
        
        y = np.asarray(y).reshape(-1)
        
        for epoch in range(self.n_iter):

            prediction = X_b @ self.weights
            error = prediction - y
            
            cost = (1/m) * np.sum(error**2)
            self.cost_history.append(cost)
            
            # J = (((X_b @ self.weights) - y) ** 2) / m ----- objective function
            
            # del_J = (2/m) * (X_b.T @ ((X_b @ self.weights) - y) )
            del_J = (2/m) * (X_b.T @ error)
            
            if np.linalg.norm(del_J) < self.tol:
                break
            
            self.weights -= self.learning_rate * del_J
            
        return self.weights
    
    def predict(self, X):
        X_b = np.column_stack((np.ones(len(X)), X))
        return X_b @ self.weights