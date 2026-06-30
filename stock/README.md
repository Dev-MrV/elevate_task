# 📈 DeepQuant: Stock Price Trend Prediction using Stacked LSTM

![Python Version](https://img.shields.io/badge/Python-v3.9+-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_v2.15+-FF6F00?style=flat-square&logo=tensorflow)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)
![Data Source](https://img.shields.io/badge/Data-Yahoo_Finance-7000?style=flat-square)

DeepQuant is an institutional-grade algorithmic prediction engine designed to forecast short-term financial market movements. By combining a multi-layered Long Short-Term Memory (LSTM) recurrent neural network with robust technical analysis indicators (SMA, RSI), the system captures complex chronological patterns within live market data to generate forward-looking asset valuations.

---

## 🏗️ System Architecture & Features Breakdown

*   **Time-Series Deep Learning:** Implements a stacked Long Short-Term Memory network tailored specifically for chronological sequences, allowing it to capture complex, non-linear market trends and momentum shifts.
*   **Algorithmic Feature Engineering:** Automatically computes and integrates critical technical indicators alongside raw price action, including the 20-day/50-day Simple Moving Averages (SMA) and the Relative Strength Index (RSI).
*   **Decoupled Architecture:** Features an asynchronous, high-performance REST API backend built for speed, feeding directly into a premium, dark-mode, trading-style HTML/JS interactive dashboard.
*   **Preventative Regularization:** Implements explicit structural safeguards—including Dropout layers and Keras EarlyStopping callbacks—to actively mitigate overfitting and ensure robust generalization on unseen data.

### ⚙️ The Execution Workflow Loop
1.  **Live Market Data Ingestion:** Asynchronously pulls historical price chains via the Yahoo Finance API.
2.  **Technical Feature Extraction:** Computes SMA and RSI momentum indicators over the historical window.
3.  **Multi-Feature MinMaxScaler:** Normalizes all numerical matrices to a strict `[0, 1]` boundary for neural stability.
4.  **60-Day Lookback Window Generation:** Slices the continuous time-series into overlapping 60-day chronological batches.
5.  **Stacked LSTM Processing:** Feeds the multi-dimensional batches through the deep learning sequential layers.
6.  **Dollar-Value Scaling Inversion:** Inverts the output probability tensors back into real-world fiat currency values.
7.  **ApexCharts Visual Delivery:** Transmits the projected trend coordinates to the frontend UI for dynamic charting.

---

## 🚀 Local Installation & Quick-Start Guide

Follow these direct steps to initialize the machine learning environment and launch the prediction engine locally.

### 1. Repository Cloning
```bash
git clone https://github.com/yourusername/deepquant.git
cd deepquant
```

### 2. Environment Sandboxing
> [!WARNING]
> Due to the massive dependency footprint of TensorFlow and Keras, you must isolate your environment using a virtual sandbox to prevent global Python conflicts.

```bash
# Create a virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
```

### 3. Dependency Resolution
```bash
pip install -r requirements.txt
```

### 4. Spinning up the Stack
First, start the FastAPI neural inference backend:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then, simply double-click the `index.html` file in your file explorer to open the interactive trading dashboard in your default web browser.

---

## 📂 Interactive Configuration & Directory Tree

```text
deepquant/
├── app.py                     # Asynchronous FastAPI routing layer
├── lstm_stock_predictor.py    # Core ML logic, feature engineering, and neural topology
├── index.html                 # Frontend glassmorphic trading UI
├── main.js                    # Client-side API ingestion and ApexCharts rendering
├── styles.css                 # Custom UI styles and Tailwind variables
├── requirements.txt           # Python dependency tree
└── README.md                  # Project documentation
```

### 🧠 Configurable Hyperparameters
The neural network's architecture can be easily tuned by modifying the constant variables at the top of the `lstm_stock_predictor.py` file to test different algorithmic theories:

| Hyperparameter | Default Value | Description |
| :--- | :--- | :--- |
| **Lookback Window Size** | `60` | The number of historical days the LSTM processes before making a prediction. |
| **Stacked Units** | `[50, 50]` | The number of distinct LSTM cell blocks initialized across the two hidden layers. |
| **Learning Rate** | `0.001` | The step size utilized by the Adam optimizer during backpropagation. |
| **Batch Size** | `32` | The number of overlapping sequences passed through the network simultaneously. |
| **Dropout Rate** | `0.2` | The percentage of random neurons disabled during training to forcefully prevent memorization. |
