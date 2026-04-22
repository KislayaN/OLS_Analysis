from sklearn.model_selection import train_test_split

class Model_Selection:
    def __init__(self, X_dataframe=None, y_dataframe=None):
        super().__init__()
        self.X_dataframe = X_dataframe.copy()
        self.y_dataframe = y_dataframe.copy()
        
    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_dataframe, self.y_dataframe, train_size=0.8, random_state=43)
        return X_train, X_test, y_train, y_test