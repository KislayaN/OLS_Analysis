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
        
    def residual_plots(self, y, y_pred, title):
        y = np.asarray(y).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        residuals = y_pred - y
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
                plt.plot(model.cost_history, label=model.__class__.__name__, alpha=0.5)
            elif hasattr(model, 'final_mse'):
                plt.axhline(y=model.final_mse, label=model.__class__.__name__, alpha=0.5, linestyle='--')
        
        plt.xlabel("Iterations")
        plt.xlim((0, 2500))
        plt.yscale('log')
        plt.ylabel("MSE")
        plt.legend()
        plt.title("CONVERGENCE PATHS BETWEEN MODELS")
        plt.grid(alpha=0.5)
        plt.show()

    def plot_train_test_comparison(self, models_dict):
        """
        models_dict should look like:
        {
            'OLS': {'train_mse': 20, 'test_mse': 25},
            'Lasso': {'train_mse': 28, 'test_mse': 30},
            ...
        }
        """
        labels = list(models_dict.keys())
        train_mses = []
        test_mses = []
        for name, m in models_dict.items():
            if 'train_mse' not in m or 'test_mse' not in m:
                raise ValueError(f"Model '{name}' missing 'train_mse' or 'test_mse' key")
            train_mses.append(m['train_mse'])
            test_mses.append(m['test_mse'])
        x = np.arange(len(labels))  # Label locations
        width = 0.35               # Width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the bars
        rects1 = ax.bar(x - width/2, train_mses, width, label='Train MSE', color='#3498db')
        rects2 = ax.bar(x + width/2, test_mses, width, label='Test MSE', color='#e74c3c')

        # Add text for labels, title and custom x-axis tick labels
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Train vs Test MSE Gap per Model')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Function to add values on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.5f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.show()