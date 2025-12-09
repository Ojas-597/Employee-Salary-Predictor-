import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("salary_predictor_model.pkl")

st.title("💼 Employee Salary Predictor")

st.write("Enter employee details below:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=90)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = st.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
gender = st.selectbox("Gender", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0)
capital_loss = st.number_input("Capital Loss", min_value=0)
hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100)
native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "China", "Cuba", "Jamaica"])

# Prediction button
if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Predicted Income: >50K")
    else:
        st.success("✅ Predicted Income: <=50K")
