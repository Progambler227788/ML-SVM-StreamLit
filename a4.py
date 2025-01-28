import streamlit as st
import joblib
import numpy as np

# Load the saved SVM model
@st.cache_resource
def load_model():
    return joblib.load('talha_atif_svm_model.pkl')

model = load_model()

# Title of the application
st.title("Interactive SVM Prediction App")
st.write("Welcome! This app uses a pre-trained SVM model to make predictions based on user input.")

# User input form
st.header("Input Features")
# Add sliders, text inputs, or number inputs based on your model's features
feature_1 = st.number_input("Enter feature 1 value (e.g., Age):", min_value=0.0, max_value=100.0, step=0.1)
feature_2 = st.number_input("Enter feature 2 value (e.g., Salary):", min_value=0.0, max_value=100000.0, step=100.0)
feature_3 = st.number_input("Enter feature 3 value (e.g., Score):", min_value=0.0, max_value=1.0, step=0.01)

# Collect inputs into a numpy array
input_features = np.array([[feature_1, feature_2, feature_3]])

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_features)
        st.success(f"The prediction result is: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
