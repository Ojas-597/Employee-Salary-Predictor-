import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict **annual salary** using experience and inferred skillset.")

# -----------------------------
# Load Model & Features
# -----------------------------
MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "model_features.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model or feature file not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

# -----------------------------
# Skill Mapping Function
# -----------------------------
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

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🧑 Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
experience_auto = max(age - 22, 0)
experience = st.sidebar.number_input("Experience (years, editable)", min_value=0, max_value=50, value=experience_auto)

occupation = st.sidebar.selectbox(
    "Occupation",
    ["Tech-support", "Prof-specialty", "Exec-managerial", "Sales", "Craft-repair", "Other-service"]
)
skills_auto = map_skills(occupation)
skills_str = st.sidebar.selectbox(
    "Skills (editable)", 
    [skills_auto, "Python SQL", "Leadership Management", "Communication CRM", "Technical", "General"]
)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Security Notes")
st.sidebar.markdown("""
- Only CSV files are allowed for batch prediction  
- Data is processed in memory; nothing is stored  
- Invalid inputs are validated to prevent errors  
""")

# -----------------------------
# Input Summary (Main Page)
# -----------------------------
st.subheader("🔎 Input Summary")
input_df = pd.DataFrame({
    "Experience": [experience],
    "Skills": [skills_str],
    "Hours-per-week": [hours_per_week]
})
st.table(input_df)

# -----------------------------
# Single Prediction
# -----------------------------
with st.container():
    st.subheader("🔮 Single Employee Prediction")
    if st.button("Predict Salary"):
        try:
            input_encoded = pd.get_dummies(input_df, columns=["Skills"], drop_first=True)
            input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
            prediction = model.predict(input_encoded)[0]
            st.success(f"💰 Predicted Annual Salary: ₹ {int(prediction):,}")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

# -----------------------------
# Batch Prediction
# -----------------------------
with st.expander("📂 Batch Salary Prediction (Upload CSV)"):
    uploaded_file = st.file_uploader(
        "Upload CSV with columns: age, occupation, hours-per-week",
        type=["csv"]
    )
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            # Validate columns
            required_cols = ["age", "occupation", "hours-per-week"]
            missing = [col for col in required_cols if col not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                batch_df["Experience"] = batch_df["age"].apply(lambda x: max(x - 22, 0))
                batch_df["Skills"] = batch_df["occupation"].apply(map_skills)
                batch_model = batch_df[["Experience", "Skills", "hours-per-week"]]
                batch_encoded = pd.get_dummies(batch_model, columns=["Skills"], drop_first=True)
                batch_encoded = batch_encoded.reindex(columns=model_features, fill_value=0)
                batch_df["Predicted_Salary"] = model.predict(batch_encoded).astype(int)

                st.success("✅ Batch prediction completed")
                st.dataframe(batch_df.head())

                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions", csv, "predicted_salaries.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Batch prediction error: {e}")
