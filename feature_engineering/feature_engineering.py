class Feature_Engineer:
    def __init__(self, X_dataframe=None, y_dataframe=None):
        super().__init__()
        self.X_dataframe = X_dataframe.copy()
        self.target = y_dataframe.copy()
        
    def perform(self):
        index_col = [col for col in self.X_dataframe.columns if len(self.X_dataframe) == self.X_dataframe[col].nunique()]
          
        self.X_dataframe = self.X_dataframe.drop(columns=index_col)
        return self.X_dataframe