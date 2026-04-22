import pandas as pd

dataset = pd.read_csv("data/Admission_Predict.csv")

class Load_data:
    def __init__(self):
        super().__init__()
        self.dataset = dataset
        
    def get_data(self):
        self.X_dataframe = self.dataset.iloc[:, :-1]
        self.target = self.dataset.iloc[:, -1]
        return self.X_dataframe, self.target