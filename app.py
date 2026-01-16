import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict **annual salary** using experience and inferred skillset.")

# -------------------------------------------------
# Load Model & Feature Columns
# -------------------------------------------------
MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "model_features.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model or feature file not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

# -------------------------------------------------
# Skill Mapping (MUST match training)
# -------------------------------------------------
def map_skills(occupation):
    if occupation in ["Tech-support", "Prof-specialty"]:
        return "Python SQL"
    elif occupation == "Exec-managerial":
        return "Leadership Management"
    elif occupation == "Sales":
        return "Communication CRM"
    elif occupation == "Craft-repair":
        return "Technical"
    else:
        return "General"

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🧑 Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Tech-support", "Prof-specialty", "Exec-managerial",
        "Sales", "Craft-repair", "Other-service"
    ]
)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)

# -------------------------------------------------
# Feature Engineering (Same as training)
# -------------------------------------------------
experience = max(age - 22, 0)
skills_str = map_skills(occupation)

# -------------------------------------------------
# Input DataFrame
# -------------------------------------------------
input_df = pd.DataFrame({
    "Experience": [experience],
    "Skills_str": [skills_str],
    "hours-per-week": [hours_per_week]
})

st.subheader("🔎 Input Summary")
st.write(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔮 Predict Salary"):
    try:
        input_encoded = pd.get_dummies(
            input_df, columns=["Skills_str"], drop_first=True
        )

        # Align with training features
        input_encoded = input_encoded.reindex(
            columns=model_features, fill_value=0
        )

        prediction = model.predict(input_encoded)[0]
        st.success(f"💰 Predicted Annual Salary: ₹ {int(prediction):,}")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# -------------------------------------------------
# Batch Prediction
# -------------------------------------------------
st.markdown("---")
st.subheader("📂 Batch Salary Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file with columns: age, occupation, hours-per-week",
    type=["csv"]
)

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Feature Engineering
        batch_df["Experience"] = batch_df["age"].apply(lambda x: max(x - 22, 0))
        batch_df["Skills_str"] = batch_df["occupation"].apply(map_skills)

        batch_df = batch_df[["Experience", "Skills_str", "hours-per-week"]]

        batch_encoded = pd.get_dummies(
            batch_df, columns=["Skills_str"], drop_first=True
        )

        batch_encoded = batch_encoded.reindex(
            columns=model_features, fill_value=0
        )

        batch_df["Predicted_Salary"] = model.predict(batch_encoded).astype(int)

        st.success("✅ Batch prediction completed")
        st.write(batch_df.head())

        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "predicted_salaries.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Batch prediction error: {e}")
