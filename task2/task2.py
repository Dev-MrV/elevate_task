import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="COVID-19 Data Analysis Dashboard",
    layout="wide"
)

st.title("🦠 COVID-19 Data Analysis Dashboard")
st.markdown("Exploratory Data Analysis using Pandas, Matplotlib, and Seaborn")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data.csv")

# ----------------------------
# Number Formatter
# ----------------------------
def format_numbers(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.0f}K'
    return f'{x:.0f}'

formatter = FuncFormatter(format_numbers)

# ----------------------------
# Dataset Overview
# ----------------------------
st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

st.dataframe(df.head())

# ----------------------------
# Summary Statistics
# ----------------------------
st.header("Summary Statistics")
st.dataframe(df.describe())

# ----------------------------
# Missing Values
# ----------------------------
st.header("Missing Values")
st.dataframe(df.isnull().sum().reset_index().rename(
    columns={"index": "Column", 0: "Missing Values"}
))

# ----------------------------
# Numeric Columns
# ----------------------------
numeric_cols = [
    'Confirmed',
    'Deaths',
    'Recovered',
    'Active',
    'Incident_Rate',
    'Case_Fatality_Ratio'
]

# ----------------------------
# Histograms
# ----------------------------
st.header("Feature Distributions")

selected_feature = st.selectbox(
    "Select Feature",
    numeric_cols
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    df[selected_feature],
    bins=20,
    kde=True,
    ax=ax
)

ax.set_title(
    f"{selected_feature} Distribution",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel(selected_feature)
ax.set_ylabel("Frequency")
ax.xaxis.set_major_formatter(formatter)

st.pyplot(fig)

# ----------------------------
# Boxplot
# ----------------------------
st.header("Boxplot Analysis")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()

ax.set_xlabel("Features")
ax.set_ylabel("Values")
ax.yaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)

st.pyplot(fig)

# ----------------------------
# Correlation Matrix
# ----------------------------
st.header("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

ax.set_title(
    "Correlation Matrix",
    fontsize=14,
    fontweight="bold"
)

st.pyplot(fig)

# ----------------------------
# Scatter Plot
# ----------------------------
st.header("Relationship Between Features")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "Select X-axis",
        numeric_cols,
        index=0
    )

with col2:
    y_feature = st.selectbox(
        "Select Y-axis",
        numeric_cols,
        index=1
    )

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x=x_feature,
    y=y_feature,
    ax=ax
)

ax.set_title(
    f"{x_feature} vs {y_feature}",
    fontsize=14,
    fontweight="bold"
)

ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

st.pyplot(fig)

# ----------------------------
# Top Countries
# ----------------------------
st.header("Top 10 Countries by Confirmed Cases")

top10 = df.sort_values(
    by="Confirmed",
    ascending=False
).head(10)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=top10,
    x="Confirmed",
    y="Country_Region",
    ax=ax
)

ax.set_title(
    "Top 10 Countries by Confirmed Cases",
    fontsize=14,
    fontweight="bold"
)

ax.xaxis.set_major_formatter(formatter)

st.pyplot(fig)

# ----------------------------
# Insights
# ----------------------------
st.header("Key Insights")

st.markdown("""
- Countries with higher confirmed cases generally report higher deaths.
- Strong positive correlation exists between Confirmed, Deaths, Recovered, and Active cases.
- Several countries appear as outliers due to extremely high case counts.
- Case Fatality Ratio varies significantly across regions.
- Recovery numbers tend to increase with confirmed infections.
""")