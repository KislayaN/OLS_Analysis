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
    
    def fit_pipeline(self):
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
        
        lasso = Lasso(fit_intercept=True, iterations=5000)
        lasso.fit(X_train, y_train)
        
        ridge = Ridge(fit_intercept=True)
        ridge.fit(X_train, y_train)
        
        linear_reg = Linear_Regression(fit_intercept=True)
        linear_reg.fit(X_train, y_train)
        
        # Predicting values
        
        ols_pred_test = ols.predict(X_test)
        ols_pred_train = ols.predict(X_train)
        ols_test_mse = ols.calculate_mse(
            prediction=ols_pred_test,
            target=y_test
        )
        ols_train_mse = ols.calculate_mse(
            prediction=ols_pred_train,
            target=y_train
        )
        
        gd_pred_test = gradient_descent_ols.predict(X_test)
        gd_pred_train = gradient_descent_ols.predict(X_train)
        gd_test_mse = gradient_descent_ols.calculate_mse(
            prediction=gd_pred_test,
            target=y_test
        )
        gd_train_mse = gradient_descent_ols.calculate_mse(
            prediction=gd_pred_train,
            target=y_train
        )
        
        lr_pred_test = linear_reg.predict(X_test)
        lr_pred_train = linear_reg.predict(X_train)
        lr_test_mse = linear_reg.calculate_mse(
            prediction=lr_pred_test,
            target=y_test
        )
        lr_train_mse = linear_reg.calculate_mse(
            prediction=lr_pred_train,
            target=y_train
        )
        
        lasso_pred_test = lasso.predict(X_test)
        lasso_pred_train = lasso.predict(X_train)
        lasso_test_mse = lasso.calculate_mse(
            prediction=lasso_pred_test,
            target=y_test
        )
        lasso_train_mse = lasso.calculate_mse(
            prediction=lasso_pred_train,
            target=y_train
        )
        
        ridge_pred_test = ridge.predict(X_test)
        ridge_pred_train = ridge.predict(X_train)
        ridge_test_mse = ridge.calculate_mse(
            prediction=ridge_pred_test,
            target=y_test
        )
        ridge_train_mse = ridge.calculate_mse(
            prediction=ridge_pred_train,
            target=y_train
        )
        
        models_lst = [ols, gradient_descent_ols, linear_reg, lasso, ridge]
        
        plotter = Plot()
        plotter.convergence_diff(models_lst)
        
        # Building MSE Distortion plot requirements
        # Should use all the models for train test gap plot
        mse_dict_models = {}
        
        mse_dict_models['OLS'] = {'train_mse': ols_train_mse, 'test_mse': ols_test_mse}
        mse_dict_models['LR'] = {'train_mse': lr_train_mse, 'test_mse': lr_test_mse}
        mse_dict_models['GD'] = {'train_mse': gd_train_mse, 'test_mse': gd_test_mse}
        mse_dict_models['RIDGE'] = {'train_mse': ridge_train_mse, 'test_mse': ridge_test_mse}
        mse_dict_models['LASSO'] = {'train_mse': lasso_train_mse, 'test_mse': lasso_test_mse}
        
        plotter.plot_train_test_comparison(models_dict=mse_dict_models)
        
        # Get residuals
        plotter.residual_plots(y_pred=ols_pred_train, y=y_train, title='Residual plot for OLS training set')