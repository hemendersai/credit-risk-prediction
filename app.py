import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline
model = joblib.load("credit_risk_model.pkl")

# App title
st.title("Credit Risk Prediction App")
st.write("Enter customer details to predict credit risk (High Risk or Low Risk)")

# User Inputs
age = st.slider("Age", 18, 75, 30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job Type", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "none"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "none"])
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=1000)
duration = st.slider("Duration (in months)", 4, 72, 12)
purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", 
                                   "business", "domestic appliances", "repairs", 
                                   "vacation/others"])

# Handle 'none' as NaN to match training data
saving_accounts = np.nan if saving_accounts == "none" else saving_accounts
checking_account = np.nan if checking_account == "none" else checking_account

# Predict button
if st.button("Predict Credit Risk"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ High Credit Risk (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Credit Risk (Probability: {1 - probability:.2f})")
