import numpy as np

class Gradient_Descent_OLS:
    def __init__(self, iterations=1000, learning_rate=None, tol=1e-6):
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
        
        for epoch in range(self.n_iter):
            prediction = X_b @ self.weights
            error = prediction - y
            
            cost = (1/m) * np.sum(error**2)
            self.cost_history.append(cost)
            
            # J = (((X_b @ self.weights) - y) ** 2) / m ----- objective function
            
            del_J = (2/m) * (X_b.T @ ((X_b @ self.weights) - y) )
            new_weights = self.weights - self.learning_rate * del_J

            if np.linalg.norm(new_weights - self.weights) < self.tol:
                self.weights = new_weights
                break
            
        return self.weights