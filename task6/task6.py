import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Model Evaluation
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"Accuracy: {accuracy:.2%}")

st.write("Confusion Matrix")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)
# Streamlit UI
st.title("Diabetes Prediction using KNN")

st.write("Enter patient details below:")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

if st.button("Predict"):
    
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    input_scaled = scaler.transform(input_data)

    prediction = knn.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Patient is likely Diabetic")
    else:
        st.success("Patient is likely Not Diabetic")