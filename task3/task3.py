import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction using Linear Regression")

# Load Dataset
df = pd.read_csv("Housing.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================
# Preprocessing
# ==========================

df["mainroad"] = df["mainroad"].map({"yes": 1, "no": 0})
df["guestroom"] = df["guestroom"].map({"yes": 1, "no": 0})
df["basement"] = df["basement"].map({"yes": 1, "no": 0})
df["hotwaterheating"] = df["hotwaterheating"].map({"yes": 1, "no": 0})
df["airconditioning"] = df["airconditioning"].map({"yes": 1, "no": 0})
df["prefarea"] = df["prefarea"].map({"yes": 1, "no": 0})

df["furnishingstatus"] = df["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ==========================
# Sidebar Inputs
# ==========================

st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area", 1000, 20000, 5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
stories = st.sidebar.slider("Stories", 1, 5, 2)

mainroad = st.sidebar.selectbox("Main Road", ["Yes", "No"])
guestroom = st.sidebar.selectbox("Guest Room", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes", "No"])

parking = st.sidebar.slider("Parking Spaces", 0, 5, 1)

prefarea = st.sidebar.selectbox("Preferred Area", ["Yes", "No"])

furnishing = st.sidebar.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

# Encoding User Input
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

furnishing_map = {
    "Furnished": 2,
    "Semi-Furnished": 1,
    "Unfurnished": 0
}

furnishing = furnishing_map[furnishing]

# Prediction
if st.sidebar.button("Predict Price"):

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishing]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: ₹ {prediction[0]:,.2f}"
    )

# ==========================
# Model Evaluation
# ==========================

st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:,.0f}")
col2.metric("MSE", f"{mse:,.0f}")
col3.metric("R² Score", f"{r2:.3f}")

# ==========================
# Actual vs Predicted Plot
# ==========================

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(y_test, y_pred)

ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

# ==========================
# Coefficients
# ==========================

st.subheader("Feature Importance")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# ==========================
# Feature Importance
# ==========================


coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coef_df)

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.bar(coef_df["Feature"], coef_df["Coefficient"])

# Add axis labels
ax2.set_xlabel("Features")
ax2.set_ylabel("Coefficient Value")

ax2.set_title("Feature Importance (Regression Coefficients)")

plt.xticks(rotation=45)

st.pyplot(fig2)