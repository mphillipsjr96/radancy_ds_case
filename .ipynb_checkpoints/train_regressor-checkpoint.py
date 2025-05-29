import pandas as pd
from xgboost import XGBRegressor
from preprocessing import preprocess_data, build_pipeline
from utils import save_model

# Load data
data = pd.read_csv("data/ds_challenge_data.csv")
data = preprocess_data(data)
data = data[data['had_conversion'] == 1]  # Only rows with conversions

# Features & Target
numerical = ['cost', 'clicks', 'impressions', 'CPC']
categorical = ['category_id', 'industry', 'publisher', 'customer_id', 'market_id']
X = data[numerical + categorical]
y = data['CPA']

# Model
reg_model = XGBRegressor()
reg_pipeline = build_pipeline(numerical, categorical, reg_model)
reg_pipeline.fit(X, y)

# Save
save_model(reg_pipeline, "models/xgb_regressor.pkl")