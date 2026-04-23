import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load_data import Load_data
from model_selection.train_test_split import Model_Selection
from preprocessing.scaler import Scaler
from feature_engineering.feature_engineering import Feature_Engineer
from approaches.normal_variant import BLUE

# Load data
data_loader = Load_data()
X_dataframe, target = data_loader.get_data()

# Split data
splitter = Model_Selection(
    X_dataframe=X_dataframe,
    y_dataframe=target
)
X_train, X_test, y_train, y_test = splitter.split()

# Feature Engineering
feat_engineer = Feature_Engineer()
X_feature_engineer = feat_engineer.perform(X_dataframe=X_dataframe)

# Centerind the data
X_feature_engineer = X_feature_engineer - X_feature_engineer.mean()
target = target - target.mean()

# Scaler 
scaler = Scaler(perform_scaling=True)
X_train, X_test, y_train, y_test = scaler.transform(X_dataframe=X_feature_engineer, y_dataframe=target)
X_scaled = scaler.X_scaled

print(f"X_scaled:\n {X_scaled.head()}")

coefs = BLUE(feature_space=X_scaled, target=target)

from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
print(f"Coefs by LR: {lr.coef_}")
print(f"Coefs by OLS: {coefs.coefficients()}")
