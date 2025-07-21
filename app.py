import streamlit as st
import pandas as pd
import joblib

st.title("🚀 Employee Salary Prediction App")

st.write("📂 Upload a CSV file with employee features (same columns as model was trained on).")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded data
        input_data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.write(input_data.head())

        # Drop 'income' if accidentally present in uploaded CSV
        if 'income' in input_data.columns:
            input_data.drop(columns=['income'], inplace=True)

        # Load the model
        model = joblib.load("model.pkl")
        st.success("✅ Model loaded successfully!")

        # Predict
        if st.button("Predict"):
            predictions = model.predict(input_data)
            input_data['Predicted Income'] = predictions
            st.write("🎯 Prediction Results:")
            st.write(input_data)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

