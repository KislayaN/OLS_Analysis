import pandas as pd

class Load_data:
    def __init__(self):
        super().__init__()
        self.dataset = None
        
    def get_data(self):
        self.dataset = pd.read_csv("data/assets/Admission_Predict.csv")
        
        self.X_dataframe = self.dataset.iloc[:, :-1]
        self.target = self.dataset.iloc[:, [-1]]
        return self.X_dataframe, self.target