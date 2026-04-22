from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self, perform_scaling=False, X_train=None, X_test=None):
        super().__init__()
        self.perform_scaling = perform_scaling
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        
    def scalar(self):
        if self.perform_scaling == False:
            return self.X_train, self.X_test
        
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        return X_train_scaled, X_test_scaled