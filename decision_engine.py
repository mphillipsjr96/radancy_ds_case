import pandas as pd
import numpy as np
from utils import load_model
from preprocessing import preprocess_data

def recommend_best_options(df, classifier_path, regressor_path):
    """
    For each (customer_id, industry, category_id), find the best market_id and publisher.
    Returns a DataFrame with recommendations.
    """
    # Load models
    classifier = load_model(classifier_path)
    regressor = load_model(regressor_path)

    numerical = ['CPC']
    categorical = ['category_id', 'industry', 'publisher', 'customer_id', 'market_id']

    df = preprocess_data(df)

    # Unique campaign settings to evaluate
    groups = df.groupby(['customer_id', 'industry', 'category_id'])
    recommendations = []

    for key, group in groups:
        best_score = -np.inf
        best_market = None
        best_publisher = None

        for (market, publisher), subset in group.groupby(['market_id', 'publisher']):
            X = subset[numerical + categorical]
            conv_prob = classifier.predict_proba(X)[:, 1].mean()

            est_cpa = regressor.predict(X).mean()
            if np.isnan(est_cpa) or est_cpa <= 1e-3:
                est_cpa = 1.0  # assign a default CPA to avoid division issues

            score = conv_prob / est_cpa

            if score > best_score:
                best_score = score
                best_market = market
                best_publisher = publisher

        recommendations.append({
            "customer_id": key[0],
            "industry": key[1],
            "category_id": key[2],
            "recommended_market_id": best_market,
            "recommended_publisher": best_publisher,
            "score": best_score
        })

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    df = pd.read_csv("data/ds_challenge_data.csv")
    result_df = recommend_best_options(
        df,
        classifier_path="models/xgb_classifier.pkl",
        regressor_path="models/xgb_regressor.pkl"
    )
    print(result_df.head())
    result_df.to_csv("data/decision_results.csv",index=False)
