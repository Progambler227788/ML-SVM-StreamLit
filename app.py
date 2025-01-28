import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved LabelEncoder, StandardScaler, and SVM Model
encode = joblib.load('label_encoder.pkl')
scaling = joblib.load('scaler.pkl')
svm_model = joblib.load('talha_atif_svm_model.pkl')

# Streamlit page title
st.title("User Data Prediction")
with st.form("prediction_form"):
    st.header("Enter User Data")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Age", min_value=18, max_value=100, value=25)
    estimated_salary = st.number_input("Estimated Salary", min_value=1000, max_value=1000000, value=50000)
    submit_button = st.form_submit_button("Predict")

if submit_button:
    
    gender_encoded = encode.transform([gender])[0]
    input_data = pd.DataFrame([[gender_encoded, age, estimated_salary]], columns=['Gender', 'Age', 'EstimatedSalary'])
    input_data[['Age', 'EstimatedSalary']] = scaling.transform(input_data[['Age', 'EstimatedSalary']])


    prediction = svm_model.predict(input_data)
    if prediction == 1:
        st.success("Prediction: The user is likely to purchase!")
    else:
        st.error("Prediction: The user is unlikely to purchase!")

# Sample Data
st.sidebar.header("Sample Data")
st.sidebar.write("""
- **Gender**: Male/Female
- **Age**: Age range from 18 to 100
- **Estimated Salary**: The estimated annual salary in USD
""")
