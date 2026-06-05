import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction using Tree-Based Models")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("heart.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

# --------------------------------------------------
# TARGET DISTRIBUTION
# --------------------------------------------------

st.subheader("Target Distribution")

fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=df, x='target', ax=ax)
ax.set_xlabel("Target")
ax.set_ylabel("Count")
ax.set_title("Heart Disease Distribution")
st.pyplot(fig)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
st.pyplot(fig)

# --------------------------------------------------
# FEATURES & TARGET
# --------------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("Model Parameters")

max_depth = st.sidebar.slider(
    "Decision Tree Depth",
    1,
    15,
    4
)

n_estimators = st.sidebar.slider(
    "Random Forest Trees",
    10,
    300,
    100
)

# --------------------------------------------------
# DECISION TREE
# --------------------------------------------------

dt = DecisionTreeClassifier(
    max_depth=max_depth,
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

# --------------------------------------------------
# MODEL ACCURACY
# --------------------------------------------------

st.subheader("Model Accuracy")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Decision Tree Accuracy",
        f"{dt_acc:.4f}"
    )

with col2:
    st.metric(
        "Random Forest Accuracy",
        f"{rf_acc:.4f}"
    )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.subheader("Model Comparison")

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_acc, rf_acc]
})

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy",
    ax=ax
)
ax.set_ylim(0,1)
st.pyplot(fig)

# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

st.subheader("Confusion Matrices")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        confusion_matrix(y_test, dt_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Decision Tree")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        confusion_matrix(y_test, rf_pred),
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax
    )

    ax.set_title("Random Forest")

    st.pyplot(fig)

# --------------------------------------------------
# DECISION TREE VISUALIZATION
# --------------------------------------------------

st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(20,10))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------
# OVERFITTING ANALYSIS
# --------------------------------------------------

st.subheader("Overfitting Analysis")

depths = range(1,16)

train_scores = []
test_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_scores.append(
        model.score(X_train, y_train)
    )

    test_scores.append(
        model.score(X_test, y_test)
    )

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    depths,
    train_scores,
    marker='o',
    label="Training Accuracy"
)

ax.plot(
    depths,
    test_scores,
    marker='s',
    label="Testing Accuracy"
)

ax.set_xlabel("Tree Depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting Analysis")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10,6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax
)

st.pyplot(fig)

st.dataframe(importance_df)

# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

st.subheader("5-Fold Cross Validation")

scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

st.write("Scores:", scores)

st.success(
    f"Mean Accuracy: {scores.mean():.4f}"
)

st.info(
    f"Standard Deviation: {scores.std():.4f}"
)

# --------------------------------------------------
# CLASSIFICATION REPORTS
# --------------------------------------------------

st.subheader("Classification Reports")

tab1, tab2 = st.tabs(
    ["Decision Tree", "Random Forest"]
)

with tab1:
    st.text(
        classification_report(
            y_test,
            dt_pred
        )
    )

with tab2:
    st.text(
        classification_report(
            y_test,
            rf_pred
        )
    )