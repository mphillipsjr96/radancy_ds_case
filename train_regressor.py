import pandas as pd
from xgboost import XGBRegressor
from preprocessing import preprocess_data, build_pipeline
from utils import save_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
data = pd.read_csv("data/ds_challenge_data.csv")
data = preprocess_data(data)
data = data[data['conversions'] > 0]  # Only rows with conversions

# Features & Target
numerical = ['CPC']
categorical = ['category_id', 'industry', 'publisher', 'customer_id', 'market_id']
X = data[numerical + categorical]
y = data['CPA']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
reg_model = XGBRegressor()
reg_pipeline = build_pipeline(numerical, categorical, reg_model)
reg_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = reg_pipeline.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Regressor Performance on Test Set:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# Model
reg_model = XGBRegressor()
reg_pipeline = build_pipeline(numerical, categorical, reg_model)
reg_pipeline.fit(X, y)

# Save
save_model(reg_pipeline, "models/xgb_regressor.pkl")