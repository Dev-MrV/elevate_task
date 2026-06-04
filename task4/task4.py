import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    roc_auc_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("🎗 Breast Cancer Diagnosis Prediction")
st.markdown("### Logistic Regression Binary Classification Project")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------
df = df.copy()

# Remove unnamed column if present
if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

# Drop ID column
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Encode diagnosis
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Model Settings")

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head())

# ---------------------------------------------------
# DATA TYPES
# ---------------------------------------------------
st.header("Column Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(info_df)

# ---------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------
st.header("Diagnosis Distribution")

fig, ax = plt.subplots(figsize=(6,4))

counts = df["diagnosis"].value_counts().sort_index()

ax.bar(
    ["Benign (0)", "Malignant (1)"],
    counts.values
)

ax.set_xlabel("Diagnosis")
ax.set_ylabel("Count")
ax.set_title("Class Distribution")

st.pyplot(fig)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------
st.header("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    ax=ax
)

ax.set_title("Feature Correlation Matrix")

st.pyplot(fig)

# ---------------------------------------------------
# FEATURE DISTRIBUTION
# ---------------------------------------------------
st.header("Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    [col for col in df.columns if col != "diagnosis"]
)

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    df[feature],
    bins=25
)

ax.set_title(f"Distribution of {feature}")
ax.set_xlabel(feature)
ax.set_ylabel("Frequency")

st.pyplot(fig)

# ---------------------------------------------------
# BOXPLOT
# ---------------------------------------------------
st.header("Boxplot")

box_feature = st.selectbox(
    "Select Feature for Boxplot",
    [col for col in df.columns if col != "diagnosis"],
    key="boxplot"
)

fig, ax = plt.subplots(figsize=(6,4))

ax.boxplot(df[box_feature])

ax.set_title(box_feature)

st.pyplot(fig)

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# STANDARDIZATION
# ---------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_train_scaled,
    y_train
)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------
y_prob = model.predict_proba(X_test_scaled)[:, 1]

y_pred = (y_prob >= threshold).astype(int)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
st.header("Model Evaluation")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{accuracy:.3f}")
c2.metric("Precision", f"{precision:.3f}")
c3.metric("Recall", f"{recall:.3f}")
c4.metric("F1 Score", f"{f1:.3f}")
c5.metric("ROC-AUC", f"{roc_auc:.3f}")

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
st.header("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5,4))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

disp.plot(ax=ax)

st.pyplot(fig)

# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------
st.header("ROC Curve")

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

fig, ax = plt.subplots(figsize=(7,5))

ax.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

ax.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

st.pyplot(fig)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
st.header("Feature Importance")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coef_df["Absolute"] = coef_df["Coefficient"].abs()

coef_df = coef_df.sort_values(
    by="Absolute",
    ascending=False
)

st.dataframe(
    coef_df[
        ["Feature", "Coefficient"]
    ]
)

fig, ax = plt.subplots(figsize=(10,8))

top_features = coef_df.head(15)

ax.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

ax.set_title("Top 15 Important Features")

st.pyplot(fig)

# ---------------------------------------------------
# THRESHOLD TUNING
# ---------------------------------------------------
st.header("Threshold Tuning")

st.write(
    f"""
Current Threshold: **{threshold:.2f}**

- Lower threshold increases Recall.
- Higher threshold increases Precision.
- Default threshold is 0.50.
"""
)

# ---------------------------------------------------
# SIGMOID FUNCTION
# ---------------------------------------------------
st.header("Sigmoid Function")

st.latex(
    r"P(y=1)=\frac{1}{1+e^{-z}}"
)

st.write("""
The sigmoid function converts any real number into a probability between 0 and 1.

If the probability is greater than the selected threshold,
the sample is classified as Malignant (1),
otherwise Benign (0).
""")

# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------
st.header("Project Summary")

st.write(f"""
Dataset Size: {df.shape[0]} rows

Number of Features: {X.shape[1]}

Test Size: {test_size}

Classification Threshold: {threshold}

Model Used: Logistic Regression

ROC-AUC Score: {roc_auc:.3f}
""")