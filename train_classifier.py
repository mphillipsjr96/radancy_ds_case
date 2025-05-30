import pandas as pd
from xgboost import XGBClassifier
from preprocessing import preprocess_data, build_pipeline
from utils import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load data
data = pd.read_csv("data/ds_challenge_data.csv")
data = preprocess_data(data)
data['had_conversion'] = (data['conversions'] > 0).astype(int)

# Features & Target
numerical = ['CPC']
categorical = ['category_id', 'industry', 'publisher', 'customer_id', 'market_id']
X = data[numerical + categorical]
y = data['had_conversion']



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model
clf_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf_pipeline = build_pipeline(numerical, categorical, clf_model)
clf_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]

# Metrics
print("Classifier Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Model
clf_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf_pipeline = build_pipeline(numerical, categorical, clf_model)
clf_pipeline.fit(X, y)

# Save
save_model(clf_pipeline, "models/xgb_classifier.pkl")
