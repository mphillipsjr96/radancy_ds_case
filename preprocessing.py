# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from utils import save_model

def preprocess_data(df):
    df = df.rename(columns={'converions': 'conversions'})
    df = df[df['cost'] > 0].copy()
    df['CPC'] = df['cost'] / df['clicks'].replace(0, np.nan)
    df['CPA'] = df['cost'] / df['conversions'].replace(0, np.nan)
    return df

def build_pipeline(numerical_features, categorical_features, model):

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add group-level average CPA features to the dataframe.
    These represent historical average CPAs for customer, industry, and category.
    """
    # Avoid leakage by excluding the current row from its own group average
    for group in ['customer_id', 'industry', 'category_id']:
        group_avg = (
            df.groupby(group)['CPA']
            .transform(lambda x: (x.sum() - x) / (x.count() - 1).clip(lower=1))
            .fillna(df['CPA'].mean())  # Handle any NaN with global mean
        )
        df[f'{group}_avg_cpa'] = group_avg

    return df
