from sklearn.model_selection import train_test_split

class Model_Selection:
    def __init__(self, X_dataframe=None, y_dataframe=None):
        super().__init__()
        self.X_dataframe = X_dataframe.copy()
        self.y_dataframe = y_dataframe.copy()
        
    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_dataframe, self.y_dataframe)
        return X_train, X_test, y_train, y_test