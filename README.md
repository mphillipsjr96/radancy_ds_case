# radancy_ds_case
 
# CPA Prediction API

This project is a lightweight FastAPI service for predicting Cost Per Acquisition (CPA) using a trained regression model. The service includes preprocessing logic, model inference, and can be containerized via Docker.

## 🚀 Features

- FastAPI backend for serving predictions
- Preprocessing with scikit-learn pipelines
- Regression model for CPA prediction
- Docker container for easy deployment

## 🧠 Model

The model is trained using historical marketing data. It takes into account features like:

- `CPC` (Cost per Click)
- `category_id`
- `industry`
- `publisher`
- `customer_id`
- `market_id`

## 🛠 Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Train the model**:
    ```bash
    python train_regressor.py
3. **🐳 Docker**:
    To build and run the container:

        docker build -t cpa-regression-api .
        docker run -p 8000:8000 cpa-regression-api
4. **Test the API**:
    ```bash
    python test_api.py
## 📬 Example Request
    POST /predict
    {
        "CPC": 10.22,
        "category_id": "291000",
        "industry": "Marketing",
        "publisher": "2b764",
        "customer_id": "746",
        "market_id": "12733593",
        "conversions": 1 #looking to find a way to not need this
    }

## 🔁 Response
    {
        'predicted_CPA': 3.1178
    }

## 📁 File Structure

    ├── data/
    │ ├── ds_challenge_data.csv # Training data
    │ └── decision_results.csv # output of the decision_engine.py
    ├── models/
    │ ├── xgb_regressor.pkl #pkl file with the trained regression model
    ├── notebooks/
    │ ├── eda.ipynb # Playground notebook
    ├── .dockerignore
    ├── api_server.py #to run the server not on Docker
    ├── decision_engine.py #A script that produces the best market_id & publisher based on customer_id, industry, and category_id
    ├── Dockerfile #
    ├── preprocessing.py #Functions that are used in the preprocessing phase
    ├── requirements.txt #Packages & versions that are used in the project
    ├── test_api.py #Used to test if the API is working
    ├── test_pipeline.py #Used to test if the pipeline is working
    ├── train_regressor.py #Used to train the CPA regression model
    ├── utils.py #Extra functions that were used

## 🧪 Notes

Make sure the model is retrained when the data schema or source changes.