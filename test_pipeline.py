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
classifier = load_model("models/xgb_classifier.pkl")
regressor = load_model("models/xgb_regressor.pkl")

# Predict conversion
conversion_preds = classifier.predict_proba(X_sample)[:, 1]  # probability of conversion
print("Conversion Probabilities:\n", conversion_preds)

# Predict CPA only where conversion is likely
likely_conversions = X_sample[conversion_preds > 0.5]
if not likely_conversions.empty:
    cpa_preds = regressor.predict(likely_conversions)
    print("\nPredicted CPAs:\n", cpa_preds)
else:
    print("\nNo likely conversions to predict CPA.")
