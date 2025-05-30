from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from preprocessing import preprocess_data
from utils import load_model

# Load model and columns
regressor = load_model("models/xgb_regressor.pkl")

app = FastAPI(title="CPA Prediction API")

# Define request schema
class CPARequest(BaseModel):
    CPC:float
    conversions: int
    category_id: str
    industry: str
    publisher: str
    customer_id: str
    market_id: str

@app.post("/predict")
def predict_cpa(input_data: CPARequest):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data.dict()])
        print(df)
        # Predict CPA
        prediction = regressor.predict(df)[0]
        print(prediction)
        return {"predicted_CPA": round(float(prediction), 4)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="localhost", port=8000, reload=True)