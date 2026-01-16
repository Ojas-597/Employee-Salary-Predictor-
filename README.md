# 💼 Employee Salary Predictor

A Machine Learning–based web application that predicts the estimated salary of an employee using experience and inferred skillsets.
The system uses Random Forest Regression and is deployed using Streamlit.

📌 Project Overview

Salary determination in organizations often depends on manual rules and subjective judgment.
This project applies machine learning techniques to predict employee salary in a data-driven and unbiased manner.

The system performs:

Feature engineering from employee attributes

Skill inference from occupation

Salary prediction using a trained ML model

Web-based interaction through Streamlit

🎯 Objectives

Predict employee salary using machine learning

Perform feature engineering using experience and skill mapping

Compare multiple regression models

Select the best-performing model

Deploy the model using a user-friendly web interface

⚙️ System Architecture

The application follows a simple pipeline:

User / HR
   ↓
Streamlit Web App
   ↓
Feature Engineering
(Experience + Skill Mapping)
   ↓
Data Preprocessing
(One-Hot Encoding)
   ↓
Random Forest Regressor
   ↓
Predicted Salary

🧪 Dataset & Feature Engineering
🔹 Features Used

Age

Occupation

Hours per Week

Experience (derived as Age − 22)

Skills (inferred from occupation)

🔹 Skill Mapping Logic
Occupation	Inferred Skills
Tech-support / Prof-specialty	Python, SQL
Exec-managerial	Leadership, Management
Sales	Communication, CRM
Craft-repair	Technical
Others	General
🔹 Target Variable

Salary (converted from income class to numeric value)

🧠 Machine Learning Models Used

The following regression models were evaluated:

Linear Regression

K-Nearest Neighbors Regressor

Support Vector Regressor (SVR)

Gradient Boosting Regressor

Random Forest Regressor (Selected)

Random Forest Regressor was chosen based on its superior R² score and ability to handle non-linear relationships.

🌐 Web Application (Streamlit)

The Streamlit app allows:

Single employee salary prediction

Batch prediction via CSV upload

Real-time prediction without storing data

No database is used; predictions are performed in memory.

🗂️ Project Structure
Employee-Salary-Predictor/
│
├── app.py                  # Streamlit application
├── best_model.pkl          # Trained ML model
├── model_features.pkl      # Feature columns
├── employee_salary.ipynb   # Training notebook
├── dataset.csv             # Training dataset
├── README.md               # Project documentation

🚀 How to Run the Project
1️⃣ Install dependencies
pip install streamlit pandas scikit-learn joblib

2️⃣ Run the Streamlit app
streamlit run app.py

🔐 Security Considerations

No persistent data storage

No user authentication

No sensitive data retention

Reduced risk of data breaches

This design aligns with cybersecurity best practices.

🧾 Conclusion

The Employee Salary Predictor demonstrates how machine learning can be applied to solve real-world HR problems.
Feature engineering and Random Forest Regression significantly improve prediction accuracy, while Streamlit enables easy deployment and interaction.

🔮 Future Enhancements

Use real salary datasets

Resume-based skill extraction

Model explainability using SHAP

Cloud deployment

📚 References

Scikit-learn Documentation

Pandas Documentation

Streamlit Documentation

 Machine Learning

👩‍🎓 Author

Ojaswita Ranjit Desai
M.Sc. CS (Cybersecurity)
Chhatrapati Shahu Institute of Business Education & Research (CSIBER)
