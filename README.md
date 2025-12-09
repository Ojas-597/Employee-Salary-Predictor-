# 💼 Employee Salary Predictor

A simple machine learning web app that predicts whether an employee earns more than $50K per year based on personal and job-related details.
It uses a trained model and a simple Streamlit web app.

---

## 📁 Project Files

- `app.py` → Streamlit web app
- `best_model.pkl` → Trained ML model
- `adult 3.csv` → Dataset used for training
- `requirements.txt` → Required Python libraries

---

## 🚀 Features

- Predicts salary category: >50K or <=50K
- Easy-to-use web interface
- Built using Python and Streamlit
- Trained using machine learning models

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## 📊 Dataset

This project uses the Adult Income (Census Income) dataset.

Features include:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Gender
- Hours per week
- Native country

---

## 🖥️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/Ojas-597/Employee-Salary-Predictor-.git

---
2. Install Requirement :

```bash
pip install -r requirements.txt

---
3.Run the app:

```bash
streamlit run app.py

---

## 🧠 How it Works

The app takes user inputs like:
Education level
Age
Job role
Experience
Then it uses the trained model to predict salary.

---

## ✅ Output

The app displays the predicted salary instantly on the webpage.

---

## 👤 Author

Created by Ojas


---

## Quick Important Fix for Your Repo ⚠️

Your file name is correct now:
✅ `best_model.pkl`

Make sure your `app.py` includes:

```python
model = joblib.load("best_model.pkl")
