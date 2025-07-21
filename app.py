import streamlit as st
import pandas as pd
import joblib

# Load the model once at the top
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# List of expected columns (must match the model training)
expected_columns = [
    'age', 'educational-num', 'hours-per-week',
    'occupation_Admin-clerical', 'occupation_Exec-managerial', 'occupation_Other-service',
    'gender_Female', 'gender_Male',
    'relationship_Not-in-family', 'relationship_Other'
]

st.title("🚀 Employee Salary Prediction App")

st.write("📂 Upload a CSV file with raw employee data, and the model will predict the income class.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded data
        input_data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.write(input_data.head())

        # Basic preprocessing
        # Encode categorical columns as per training (this assumes limited known values)
        df = input_data.copy()

        # Rename columns to match model expectation (if needed)
        df.rename(columns={
            'education-num': 'educational-num',
            'sex': 'gender'
        }, inplace=True)

        # One-hot encoding for 'occupation', 'gender', and 'relationship'
        df = pd.get_dummies(df, columns=['occupation', 'gender', 'relationship'])

        # Add any missing expected columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default 0

        # Reorder columns to match model
        df = df[expected_columns]

        st.success("✅ Data processed and aligned with model!")

        if st.button("Predict"):
            predictions = model.predict(df)
            input_data['Predicted Income'] = predictions
            st.write("🎯 Prediction Results:")
            st.write(input_data)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
