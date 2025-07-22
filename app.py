import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("🚀 Employee Salary Prediction App")

# Toggle input method
option = st.radio("Choose input method:", ["Upload CSV", "Manual Input"])

# ------------------------------------
# Option 1: Upload CSV and Predict
# ------------------------------------
if option == "Upload CSV":
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

            # Load model
            model = joblib.load("model.pkl")
            st.success("✅ Model loaded successfully!")

            if st.button("Predict"):
                predictions = model.predict(input_data)
                input_data['Predicted Income'] = predictions
                st.write("🎯 Prediction Results:")
                st.write(input_data)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ------------------------------------
# Option 2: Manual Input
# ------------------------------------
elif option == "Manual Input":
    st.subheader("📝 Enter Employee Details")

    # Manual inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Doctorate'])
    educational_num = st.number_input("Educational-Num", min_value=1, max_value=20, value=10)
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Married-spouse-absent', 'Widowed'])
    occupation = st.selectbox("Occupation", ['Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Adm-clerical', 'Sales', 'Machine-op-inspct'])
    relationship = st.selectbox("Relationship", ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India'])
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    # Create single-row DataFrame
    input_dict = {
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'educational-num': [educational_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss]
    }

    input_df = pd.DataFrame(input_dict)

    if st.button("Predict Income"):
        try:
            # Encode categorical features
            label_encoders = {}
            for col in input_df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                input_df[col] = le.fit_transform(input_df[col])
                label_encoders[col] = le

            # Load model
            model = joblib.load("model.pkl")
            st.success("✅ Model loaded successfully!")

            # Predict
            prediction = model.predict(input_df)
            st.write(f"🎯 **Predicted Income:** `{prediction[0]}`")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")



