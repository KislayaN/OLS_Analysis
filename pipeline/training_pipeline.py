import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load_data import Load_data
from model_selection.train_test_split import Model_Selection
from preprocessing.scaler import Scaler
from feature_engineering.feature_engineering import Feature_Engineer
from approaches.normal_variant import OLS
from approaches.Gradient_Descent import Gradient_Descent_OLS
from models.linear_regression import Linear_Regression
from models.lasso import Lasso
from models.ridge import Ridge

from plots.plots import Plot

class Pipeline:
    def __init__(self):
        pass
    
    def fit_pipline(self):
        # Load data
        data_loader = Load_data()
        X_dataframe, target = data_loader.get_data()

        # Feature Engineering
        feat_engineer = Feature_Engineer()
        X_feature_engineer = feat_engineer.fit(X_dataframe=X_dataframe)
        
        # Correlation plot before introducing multicollinearity
        # plotter = Plot()
        # plotter.plot_corr(
        #     X_dataframe=X_feature_engineer,
        #     y_dataframe=target
        # )
        
        X_feature_engineer_new = X_feature_engineer.copy()
        X_feature_engineer_new['correlated_with_CGPA'] = 1.3 * X_feature_engineer_new['CGPA'] 
        
        # Correlation plot after introducing multicollinearity
        # plotter = Plot()
        # plotter.plot_corr(
        #     X_dataframe=X_feature_engineer_new,
        #     y_dataframe=target
        # )

        # Split data
        splitter = Model_Selection(
            X_dataframe=X_feature_engineer,
            y_dataframe=target
        )
        X_train, X_test, y_train, y_test = splitter.split()

        # Scaler 
        scaler = Scaler(perform_scaling=True)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Fitting Models

        ols = OLS(fit_intercept=True)
        ols.fit(X_train, y_train)
        
        gradient_descent_ols = Gradient_Descent_OLS(
            learning_rate=0.001, 
            iterations=5000,
            fit_intercept=True)
        gradient_descent_ols.fit(X_train, y_train)
        
        lasso = Lasso(fit_intercept=True)
        lasso.fit(X_train, y_train)
        
        ridge = Ridge(fit_intercept=True)
        ridge.fit(X_train, y_train)
        
        linear_reg = Linear_Regression(fit_intercept=True)
        linear_reg.fit(X_train, y_train)
        
        # Predicting values
        
        ols_pred = ols.predict(X_test)
        ols.calculate_mse(
            prediction=ols_pred,
            target=y_test
        )
        
        gradient_descent_ols_pred = gradient_descent_ols.predict(X_test)
        gradient_descent_ols.calculate_mse(
            prediction=gradient_descent_ols_pred,
            target=y_test
        )
        
        linear_reg_pred = linear_reg.predict(X_test)
        linear_reg.calculate_mse(
            prediction=linear_reg_pred,
            target=y_test
        )
        
        lasso_pred = lasso.predict(X_test)
        lasso.calculate_mse(
            prediction=lasso_pred,
            target=y_test
        )
        
        ridge_pred = ridge.predict(X_test)
        ridge.calculate_mse(
            prediction=ridge_pred,
            target=y_test
        )
        
        models_lst = [ols, gradient_descent_ols, linear_reg, lasso, ridge]
        
        plotter = Plot()
        plotter.convergence_diff(models_lst)
        
        # Should use all the models for train test gap plot
        
pipeline = Pipeline()
pipeline.fit_pipline()