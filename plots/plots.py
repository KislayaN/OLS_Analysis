import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import t

class Plot:
    def __init__(self):
        super().__init__()
    
    def plot_corr(self, X_dataframe, y_dataframe):
        dataframe = pd.concat(X_dataframe, y_dataframe, axis=1)
        
        corr_matrix = dataframe.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.show()
        
    def coefficient_plot(self, X, y_pred, y_true, coefficients):
        X = X.copy()
        feature_names = X.columns.tolist()
        
        residuals = np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1)
        
        # Computing variance 
        n = X.shape[0]
        p = X.shape[1]
        
        sigma_2 = (residuals @ residuals) / (n - p)
        
        # Compute covariance matrix
        covariance_beta = sigma_2 * np.linalg.inv(X.T @ X)
        
        # Get standard Error
        standard_error = np.sqrt(np.diag(covariance_beta))
        
        # Confidence Intervals
        lower = coefficients - 1.96 * standard_error
        upper = coefficients + 1.96 * standard_error
        
        df = pd.DataFrame({
            "feature": feature_names,
            "coef": coefficients,
            "lower": lower,
            "upper": upper
        })
        
        plt.errorbar(
            x=df["coef"],
            y=df["feature"],
            xerr=[
                df["coef"] - df["lower"],
                df["upper"] - df["coef"]
            ],
            fmt='o',
            capsize=4
        )

        plt.axvline(0, linestyle='--')  # zero line
        plt.xlabel("Coefficient")
        plt.ylabel("Features")
        plt.title("Coefficient Plot")

        plt.show()
        
    def residual_plots(self, y_pred, residuals, title):
        plt.scatter(y_pred, residuals)
        plt.axhline(y = 0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(title)
        plt.show()