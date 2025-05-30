import pandas as pd
from preprocessing import preprocess_data
from utils import load_model

# Load sample data
data = pd.read_csv("data/ds_challenge_data.csv")
data = preprocess_data(data)
sample = data.sample(5, random_state=42)  # grab a few rows

numerical = ['CPC']
categorical = ['category_id', 'industry', 'publisher', 'customer_id', 'market_id']
X_sample = sample[numerical + categorical]

# Load models
regressor = load_model("models/xgb_regressor.pkl")

# Predict CPA only where conversion is likely
cpa_preds = regressor.predict(X_sample)
print("\nPredicted CPAs:\n", cpa_preds)

