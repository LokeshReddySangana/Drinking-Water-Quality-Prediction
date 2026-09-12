# ============================================================
# Drinking Water Quality Prediction Using Machine Learning
# Streamlit App (Dataset Upload Version)
# ============================================================

import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Drinking Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

with open("water_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

st.title("💧 Drinking Water Quality Prediction")

st.write("""
Upload a CSV file containing water quality parameters.
The model will predict whether each water sample is safe for drinking.
""")

# ------------------------------------------------------------
# Upload Dataset
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Water Quality Dataset (.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    # Remove target column if present
    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
    else:
        X = df

    # Prediction
    prediction = model.predict(X)
    probability = model.predict_proba(X)

    # Add Prediction Column
    df["Prediction"] = [
        "Safe" if p == 1 else "Unsafe"
        for p in prediction
    ]

    df["Confidence (%)"] = (
        probability.max(axis=1) * 100
    ).round(2)

    st.subheader("Prediction Results")

    st.dataframe(df)

    # Summary
    safe = (prediction == 1).sum()
    unsafe = (prediction == 0).sum()

    col1, col2 = st.columns(2)

    col1.metric("Safe Water Samples", safe)
    col2.metric("Unsafe Water Samples", unsafe)

    # Download Button
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="water_predictions.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# Show Project Graphs
# ------------------------------------------------------------

st.markdown("---")

st.subheader("Project Visualizations")

graphs = [
    ("Missing Values", "graphs/missing_values.png"),
    ("Class Distribution", "graphs/class_distribution.png"),
    ("Correlation Heatmap", "graphs/correlation_heatmap.png"),
    ("Accuracy Chart", "graphs/accuracy_chart.png"),
    ("Confusion Matrix", "graphs/confusion_matrix.png"),
    ("Feature Importance", "graphs/feature_importance.png")
]

for title, path in graphs:
    if os.path.exists(path):
        st.write(f"### {title}")
        st.image(path, use_container_width=True)

st.markdown("---")
st.caption("Developed using Python, Scikit-learn and Streamlit")