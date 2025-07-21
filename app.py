import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("🚀 Employee Salary Prediction App")

st.write("📂 Upload a CSV file with employee features (raw values).")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.write(input_data.head())

        # Drop income if present
        if 'income' in input_data.columns:
            input_data.drop(columns=['income'], inplace=True)

        # Encode categorical features
        label_encoders = {}
        for col in input_data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            input_data[col] = le.fit_transform(input_data[col])
            label_encoders[col] = le

        # Load the model
        model = joblib.load("model.pkl")
        st.success("✅ Model loaded successfully!")

        if st.button("Predict"):
            predictions = model.predict(input_data)
            input_data['Predicted Income'] = predictions
            st.write("🎯 Prediction Results:")
            st.write(input_data)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")


