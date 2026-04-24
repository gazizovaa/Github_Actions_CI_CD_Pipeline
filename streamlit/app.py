import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import os 

# Configure the main page
st.set_page_config(page_title='Medical Insurance Costs Prediction', layout='centered')

# Load the model
@st.cache_resource
def load_predicted_model():
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file is not found.")
        return None 

model = load_predicted_model()

# Design an interface of the application
st.title("Medical Insurance Costs Calculation")

# Create columns to add user inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(label="Age", min_value=1, max_value=100, value=25)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input(label="BMI", min_value=10.0, max_value=60.0, value=25.0)
    
with col2:
    children = st.number_input(label="The number of children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Are you smoking?", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
# Put the button for prediction
if st.button("Calculate costs"):
    if model:
        # Convert the input data into DataFrame as the model expects
        input_data = pd.DataFrame({"age": [age], "sex": [sex], "bmi": [bmi],
                                   "children": [children], "smoker": [smoker], 
                                   "region": [region]})
        log_prediction = model.predict(input_data)
        
        # Use np.expm1 to convert from log into real values
        real_values = np.expm1(log_prediction)[0]
        st.success(f"{real_values:,.2f}")

st.sidebar.header("About Dataset")
st.sidebar.info("""
                The medical insurance dataset encompasses various factors influencing medical expenses, such as age, sex, BMI, smoking status, number of children, and region. This dataset serves as a foundation for training machine learning models capable of forecasting medical expenses for new policyholders.
                Its purpose is to shed light on the pivotal elements contributing to increased insurance costs, aiding the company in making more informed decisions concerning pricing and risk assessment.
                """)