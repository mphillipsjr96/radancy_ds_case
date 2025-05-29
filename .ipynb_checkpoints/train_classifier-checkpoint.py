import pandas as pd
from xgboost import XGBClassifier
from preprocessing import preprocess_data, build_pipeline
from utils import save_model

# Load data
data = pd.read_csv("data/ds_challenge_data.csv")
data = preprocess_data(data)

# Features & Target
numerical = ['cost', 'clicks', 'impressions', 'CPC']
categorical = ['category_id', 'industry', 'publisher', 'customer_id', 'market_id']
X = data[numerical + categorical]
y = data['had_conversion']

# Model
clf_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf_pipeline = build_pipeline(numerical, categorical, clf_model)
clf_pipeline.fit(X, y)

# Save
save_model(clf_pipeline, "models/xgb_classifier.pkl")
