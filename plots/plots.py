import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Plot:
    def __init__(self):
        super().__init__()
    
    def plot_corr(self, X_dataframe, y_dataframe):
        dataframe = pd.concat(X_dataframe, y_dataframe, axis=1)
        
        corr_matrix = dataframe.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.show()