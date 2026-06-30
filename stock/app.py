from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio

from lstm_stock_predictor import (
    fetch_data,
    advanced_feature_engineering,
    preprocess_data_advanced,
    build_model,
    train_model,
    evaluate_and_return_data
)

app = FastAPI(title="LSTM Quant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ticker: str
    lookback: int = 60
    epochs: int = 50
    years: int = 5

@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Executes the full LSTM quant pipeline and returns the data payload."""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=request.years * 365)).strftime('%Y-%m-%d')
        
        df_raw = fetch_data(request.ticker.upper(), start_date, end_date)
        df_engineered = advanced_feature_engineering(df_raw)
        
        X_train, y_train, X_val, y_val, scaler_target, val_dates = preprocess_data_advanced(
            df_engineered, lookback=request.lookback, apply_pca=True
        )
        
        input_shape = (X_train.shape[1], X_train.shape[2]) 
        model = build_model(input_shape)
        
        # Offload blocking training process to a separate thread
        await asyncio.to_thread(
            train_model, model, X_train, y_train, X_val, y_val, request.epochs, 32
        )
        
        payload = evaluate_and_return_data(model, X_val, y_val, scaler_target, val_dates, df_engineered)
        return payload
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
