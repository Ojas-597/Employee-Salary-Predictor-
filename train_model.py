import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("adult 3.csv")

print("✅ Columns found:", data.columns)

# -----------------------------
# Required columns
# -----------------------------
required_cols = [
    "age",
    "occupation",
    "hours-per-week",
    "income"
]

missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise Exception(f"❌ Missing columns in CSV: {missing_cols}")

data = data[required_cols]

# -----------------------------
# Feature Engineering
# -----------------------------

# Experience derived from age
data["Experience"] = data["age"] - 22
data["Experience"] = data["Experience"].clip(lower=0)

# Skill mapping from occupation
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

data["Skills_str"] = data["occupation"].apply(map_skills)

# -----------------------------
# Income to Salary conversion
# -----------------------------
def income_to_salary(income):
    if income.strip() == ">50K":
        return 90000
    else:
        return 40000

data["Salary"] = data["income"].apply(income_to_salary)

# -----------------------------
# Drop unused columns
# -----------------------------
data.drop(columns=["age", "occupation", "income"], inplace=True)

# -----------------------------
# One-Hot Encoding
# -----------------------------
data_encoded = pd.get_dummies(data, columns=["Skills_str"], drop_first=True)

# -----------------------------
# Features & Target
# -----------------------------
X = data_encoded.drop(columns=["Salary"])
y = data_encoded["Salary"]

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Save model & feature columns
# -----------------------------
joblib.dump(model, "best_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_features.pkl")

print("✅ RandomForestRegressor trained successfully")
print("✅ Model saved as best_model.pkl")
print("✅ Feature columns saved as model_features.pkl")
