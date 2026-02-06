import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HR Attrition Forecaster", layout="wide")

st.title("Employee Attrition Forecaster ")

# Load model
model = joblib.load("models/attrition_rf.pkl")

# File upload
uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocessing (same as training)
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    for col in df.select_dtypes(include="object").columns:
        if col != "Attrition":
            df[col] = df[col].astype("category").cat.codes

    if "Attrition" in df.columns:
        X = df.drop("Attrition", axis=1)
    else:
        X = df

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]
    df["Attrition_Probability"] = probs

    st.subheader("Attrition Risk Scores")
    st.dataframe(df.sort_values("Attrition_Probability", ascending=False).head(10))

    # ROI Simulation
    st.subheader("💰 Retention ROI Simulator")
    avg_cost = st.number_input("Average Replacement Cost (₹)", value=1000000)
    retention_rate = st.slider("Expected Retention Success (%)", 0, 100, 20)

    predicted_leavers = (df["Attrition_Probability"] > 0.7).sum()
    savings = predicted_leavers * avg_cost * (retention_rate / 100)

    st.metric("High-Risk Employees", predicted_leavers)
    st.metric("Estimated Annual Savings (₹)", f"{savings:,.0f}")
