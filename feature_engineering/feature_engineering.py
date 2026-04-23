class Feature_Engineer:
    def __init__(self):
        super().__init__()
        
    def fit(self, X_dataframe):
        X_dataframe = X_dataframe.copy()
        index_col = [col for col in X_dataframe.columns if len(X_dataframe) == X_dataframe[col].nunique()]
          
        self.X_dataframe = X_dataframe.drop(columns=index_col)
        return self.X_dataframe