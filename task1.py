import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Page Config
st.set_page_config(
    page_title="Data Cleaning Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Data Cleaning & Preprocessing Pipeline")
st.write("Upload a CSV file and prepare it for Machine Learning.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # Dataset Preview
        st.header("Dataset Preview")
        st.dataframe(df.head())

        # Dataset Shape
        rows, cols = df.shape

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", rows)

        with col2:
            st.metric("Columns", cols)

        # Dataset Information
        st.header("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(info_df)

        # Missing Values
        st.header("Missing Value Analysis")

        missing_values = df.isnull().sum()

        if missing_values.sum() > 0:

            missing_df = missing_values[missing_values > 0]

            st.dataframe(missing_df)

            fig, ax = plt.subplots(figsize=(8, 4))
            missing_df.plot(kind="bar", ax=ax)
            ax.set_title("Missing Values by Column")
            st.pyplot(fig)

        else:
            st.success("No missing values found.")

        # Clean Dataset
        if st.button("Clean Dataset"):

            cleaned_df = df.copy()

            # Fill Numeric Missing Values
            numeric_cols = cleaned_df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(
                    cleaned_df[col].mean()
                )

            # Fill Categorical Missing Values
            categorical_cols = cleaned_df.select_dtypes(
                include="object"
            ).columns

            for col in categorical_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].mode()[0]
                    )

            # Remove Duplicates
            duplicates = cleaned_df.duplicated().sum()
            cleaned_df.drop_duplicates(inplace=True)

            st.success(
                f"Dataset cleaned successfully. Removed {duplicates} duplicate rows."
            )

            st.session_state.cleaned_df = cleaned_df

        # Show Cleaned Dataset
        if "cleaned_df" in st.session_state:

            cleaned_df = st.session_state.cleaned_df

            st.header("Cleaned Dataset")
            st.dataframe(cleaned_df.head())

            # Encode Categorical Columns
            if st.button("Encode Categorical Columns"):

                encoded_df = cleaned_df.copy()

                cat_cols = encoded_df.select_dtypes(
                    include="object"
                ).columns

                for col in cat_cols:
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(
                        encoded_df[col].astype(str)
                    )

                st.session_state.encoded_df = encoded_df

                st.success("Categorical columns encoded successfully.")

        # Show Encoded Dataset
        if "encoded_df" in st.session_state:

            encoded_df = st.session_state.encoded_df

            st.header("Encoded Dataset")
            st.dataframe(encoded_df.head())

            # Scale Numerical Columns
            if st.button("Scale Numerical Features"):

                scaled_df = encoded_df.copy()

                numeric_cols = scaled_df.select_dtypes(
                    include=np.number
                ).columns

                scaler = StandardScaler()

                scaled_df[numeric_cols] = scaler.fit_transform(
                    scaled_df[numeric_cols]
                )

                st.session_state.scaled_df = scaled_df

                st.success("Numerical features scaled successfully.")

        # Final Dataset
        final_df = None

        if "scaled_df" in st.session_state:
            final_df = st.session_state.scaled_df

        elif "encoded_df" in st.session_state:
            final_df = st.session_state.encoded_df

        elif "cleaned_df" in st.session_state:
            final_df = st.session_state.cleaned_df

        if final_df is not None:

            st.header("Final Dataset Preview")
            st.dataframe(final_df.head())

            # Correlation Heatmap
            st.header("Correlation Heatmap")

            numeric_df = final_df.select_dtypes(
                include=np.number
            )

            if numeric_df.shape[1] > 1:

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.heatmap(
                    numeric_df.corr(),
                    annot=True,
                    cmap="coolwarm",
                    ax=ax
                )

                st.pyplot(fig)

            else:
                st.warning(
                    "Not enough numeric columns for correlation analysis."
                )

            # Distribution Plot
            st.header("Distribution Plot")

            numeric_cols = list(
                final_df.select_dtypes(
                    include=np.number
                ).columns
            )

            if len(numeric_cols) > 0:

                selected_col = st.selectbox(
                    "Select Numeric Column",
                    numeric_cols
                )

                fig, ax = plt.subplots(figsize=(8, 4))

                sns.histplot(
                    final_df[selected_col],
                    kde=True,
                    ax=ax
                )

                st.pyplot(fig)

            # Download Dataset
            st.header("Download Cleaned Dataset")

            csv = final_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file to begin.")