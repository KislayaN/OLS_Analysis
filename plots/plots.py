import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Plot:
    def __init__(self):
        super().__init__()
    
    def plot_corr(self, X_dataframe, y_dataframe):
        dataframe = pd.concat((X_dataframe, y_dataframe), axis=1)
        
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
        
    def compare_mse(self, Gradient_descent_mse,
                          Linear_regression_mse,
                          OLS_mse,
                          Ridge_mse,
                          Lasso_mse):
        plt.plot(Gradient_descent_mse, label='MSE_GD', linestyle='--')
        plt.plot(Linear_regression_mse, label='MSE_LR')
        plt.plot(OLS_mse, label='MSE_OLS')
        plt.plot(Ridge_mse, label='MSE_RIDGE')
        plt.plot(Lasso_mse, label='MSE_LASSO')
        plt.title("Comparing MSEs")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.plot()
        
    def convergence_diff(self, models):
        if not isinstance(models, list):
            raise TypeError(f"Expected a list, but recieved {type(models).__name__}")
        
        for model in models:
            if hasattr(model, 'cost_history'):
                plt.plot(model.cost_history, label=f"MSE ({model.__class__.__name__}): {model.cost_history[-1]}", alpha=0.5)
            elif hasattr(model, 'final_mse'):
                plt.axhline(y=model.final_mse, label=f"MSE ({model.__class__.__name__}): {model.final_mse:.5f}", alpha=0.5)
        
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.legend()
        plt.title("Convergence between Models based on MSE")
        plt.grid(alpha=0.5)
        plt.show()