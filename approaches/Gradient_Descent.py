import numpy as np

class Gradient_Descent:
    def __init__(self, alpha=None, epochs=None, feature_space=None, target=None, coefs=None):
        super().__init__()
        self.learning_rate = alpha
        self.epochs = epochs
        self.features = feature_space
        self.target = target
        self.n_samples = len(feature_space)
        self.coefs = coefs
        self.coefs_history = []
        
    def compute_coefficiets(self):
        for epoch in range(self.epochs):
            J = (((self.features * self.coefs) - self.target) ** 2) / self.n_samples
            
            del_J = 2 * (np.transpose(self.features) * ((self.features * self.coefs) - self.target) )
            self.coefs = self.coefs - self.learning_rate * del_J
            
            self.coefs_history = self.coefs_history.append(self.coefs)
            
        return self.coefs