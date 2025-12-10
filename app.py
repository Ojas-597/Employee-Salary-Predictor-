import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Safely load the trained model
# -----------------------------
MODEL_PATH = "best_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

st.title("🪪 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on input features.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# -----------------------------
# Create input DataFrame
# -----------------------------
input_df = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "education": [education],
    "occupation": [occupation],
    "hours-per-week": [hours_per_week],
    "experience": [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# -----------------------------
# Encode categorical features
# -----------------------------
def encode_features(df):
    for col in ["gender", "education", "occupation"]:
        df[col] = df[col].astype("category").cat.codes
    return df

encoded_input = encode_features(input_df.copy())

# ✅ Enforce exact column order used during training
expected_cols = [
    "age",
    "gender",
    "education",
    "occupation",
    "hours-per-week",
    "experience"
]

encoded_input = encoded_input[expected_cols]

# -----------------------------
# Single prediction
# -----------------------------
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(encoded_input)

        if prediction[0] in [1, ">50K"]:
            st.success("✅ Prediction: Income is >50K")
        else:
            st.success("✅ Prediction: Income is ≤50K")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# -----------------------------
# Batch prediction
# -----------------------------
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.write(batch_data.head())

    # Encode categorical columns in batch data
    for col in ["gender", "education", "occupation"]:
        if col in batch_data.columns:
            batch_data[col] = batch_data[col].astype("category").cat.codes

    try:
        batch_preds = model.predict(batch_data)
        batch_data["PredictedClass"] = batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name="predicted_classes.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Batch prediction error: {e}")

