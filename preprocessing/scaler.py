from sklearn.preprocessing import StandardScaler
import pandas as pd

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_selection.train_test_split import Model_Selection

class Scaler:
    def __init__(self, perform_scaling=False):
        super().__init__()
        self.perform_scaling = perform_scaling
        self.X_scaled = None
        
    def transform(self, X_dataframe, y_dataframe):
        X_dataframe = X_dataframe.copy()
        if self.perform_scaling == False:
            return X_dataframe
        
        nunique = X_dataframe.nunique()
        scalable_cols = [key for key in nunique.keys() if nunique[key] > 10 and key not in ['Serial No.']]
        
        splitter = Model_Selection(X_dataframe=X_dataframe, y_dataframe=y_dataframe)
        X_train, X_test, y_train, y_test = splitter.split()
        
        scaler = StandardScaler()
        
        train_cols_scaled = scaler.fit_transform(X_train[scalable_cols])
        test_cols_scaled = scaler.transform(X_test[scalable_cols])
        
        X_train[scalable_cols] = train_cols_scaled
        X_test[scalable_cols] = test_cols_scaled
        
        self.X_scaled = pd.concat((X_train, X_test), axis=0)
        
        return X_train, X_test, y_train, y_test