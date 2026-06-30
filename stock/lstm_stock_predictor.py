import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical price data from Yahoo Finance."""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    return df

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and advanced features."""
    print("Engineering advanced features (ATR, Bollinger Bands, MACD, Log Returns, Sentiment, SMA, RSI)...")
    data = df.copy()
    
    # Calculate daily log returns for stationarity
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate 21-day rolling volatility
    data['Rolling_Vol_21'] = data['Log_Returns'].rolling(window=21).std()
    
    # Volatility Metrics (ATR & Bollinger Bands)
    indicator_atr = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['ATR_14'] = indicator_atr.average_true_range()
    
    indicator_bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = indicator_bb.bollinger_hband()
    data['BB_Low'] = indicator_bb.bollinger_lband()
    data['BB_Pband'] = indicator_bb.bollinger_pband()
    
    # Momentum Metrics (MACD)
    indicator_macd = ta.trend.MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD_Line'] = indicator_macd.macd()
    data['MACD_Signal'] = indicator_macd.macd_signal()
    data['MACD_Hist'] = indicator_macd.macd_diff()
    
    # Synthetic sentiment score (-1 to 1) for demonstration
    np.random.seed(42)
    data['Sentiment_Score'] = np.random.uniform(-1, 1, size=len(data))
    
    # Standard indicators for frontend visualization
    data['SMA_20'] = ta.trend.SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data['SMA_50'] = ta.trend.SMAIndicator(close=data['Close'], window=50).sma_indicator()
    data['RSI_14'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    
    # Remove rows with NaN values caused by rolling windows
    data = data.dropna()
    
    return data

def preprocess_data_advanced(df: pd.DataFrame, lookback: int = 60, apply_pca: bool = True, pca_variance: float = 0.95):
    """Normalize features, optionally apply PCA, and generate time-series sequences."""
    print(f"Preprocessing data (lookback={lookback} days, PCA={apply_pca})...")
    
    target = df[['Close']].values
    
    feature_cols = [
        'Log_Returns', 'Rolling_Vol_21', 'ATR_14', 
        'BB_High', 'BB_Low', 'BB_Pband', 
        'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'Sentiment_Score'
    ]
    features = df[feature_cols].values
    
    # Scale target and features independently
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    target_scaled = scaler_target.fit_transform(target)
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_features.fit_transform(features)
    
    # Apply PCA for dimensionality reduction if requested
    if apply_pca:
        pca = PCA(n_components=pca_variance, random_state=42)
        features_final = pca.fit_transform(features_scaled)
        print(f"PCA reduced dimensions from {features_scaled.shape[1]} to {features_final.shape[1]}")
    else:
        features_final = features_scaled
    
    # Create sliding windows for LSTM input
    X, y = [], []
    for i in range(lookback, len(features_final)):
        X.append(features_final[i-lookback:i, :]) 
        y.append(target_scaled[i, 0])            
        
    X, y = np.array(X), np.array(y)
    
    # Chronological train/validation split (80/20)
    split_idx = int(len(X) * 0.8)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    val_dates = df.index[lookback + split_idx:]
    
    return X_train, y_train, X_val, y_val, scaler_target, val_dates

def build_model(input_shape: tuple) -> Sequential:
    """Construct a stacked LSTM neural network."""
    print(f"Building LSTM model with input shape {input_shape}...")
    model = Sequential()
    
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_model(model: Sequential, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the model using early stopping to prevent overfitting."""
    print(f"Training model for up to {epochs} epochs...")
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return history

def evaluate_and_return_data(model: Sequential, X_val, y_val, scaler_target, val_dates, df_full: pd.DataFrame):
    """Evaluate model accuracy and structure all data for the API response."""
    print("\nEvaluating model and packaging data...")
    
    y_pred_scaled = model.predict(X_val)
    
    # Inverse transform predictions back to original price scale
    y_val_actual = scaler_target.inverse_transform(y_val.reshape(-1, 1))
    y_pred_actual = scaler_target.inverse_transform(y_pred_scaled)
    
    rmse = float(np.sqrt(mean_squared_error(y_val_actual, y_pred_actual)))
    mape = float(mean_absolute_percentage_error(y_val_actual, y_pred_actual))
    
    # Determine basic buy/sell/hold signal based on 1% threshold
    last_actual_price = float(df_full['Close'].iloc[-1])
    next_predicted_price = float(y_pred_actual[-1][0])
    
    if next_predicted_price > last_actual_price * 1.01:
        direction_signal = "BUY"
    elif next_predicted_price < last_actual_price * 0.99:
        direction_signal = "SELL"
    else:
        direction_signal = "HOLD"
    
    # Replace NaNs with None for valid JSON serialization
    df_clean = df_full.replace({np.nan: None})
    
    historical_dates = df_clean.index.strftime('%Y-%m-%d').tolist()
    validation_dates = val_dates.strftime('%Y-%m-%d').tolist()
    
    predicted_prices = [float(x[0]) for x in y_pred_actual]
    aligned_predicted_prices = [None] * (len(historical_dates) - len(validation_dates)) + predicted_prices
    
    payload = {
        "metrics": {
            "rmse": rmse,
            "mape": mape,
            "direction_signal": direction_signal,
            "current_price": last_actual_price,
            "predicted_next_price": next_predicted_price
        },
        "charts": {
            "dates": historical_dates,
            "actual_prices": df_clean['Close'].tolist(),
            "predicted_prices": aligned_predicted_prices,
            "volume": df_clean['Volume'].tolist(),
            "sma20": df_clean['SMA_20'].tolist(),
            "sma50": df_clean['SMA_50'].tolist(),
            "rsi": df_clean['RSI_14'].tolist()
        }
    }
    
    return payload

def main():
    parser = argparse.ArgumentParser(description='Advanced LSTM Stock Price Predictor')
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--years', type=int, default=10)
    parser.add_argument('--no-pca', action='store_true')
    args = parser.parse_args()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.years*365)).strftime('%Y-%m-%d')
    
    print(f"--- Starting Advanced Pipeline for {args.ticker} ---")
    
    df_raw = fetch_data(args.ticker, start_date, end_date)
    df_engineered = advanced_feature_engineering(df_raw)
    
    X_train, y_train, X_val, y_val, scaler_target, val_dates = preprocess_data_advanced(
        df_engineered, lookback=60, apply_pca=not args.no_pca
    )
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train, X_val, y_val)
    
    payload = evaluate_and_return_data(model, X_val, y_val, scaler_target, val_dates, df_engineered)
    
    import json
    with open('output_payload.json', 'w') as f:
        json.dump(payload, f)
        
    print("--- Advanced Pipeline Completed Successfully ---")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
