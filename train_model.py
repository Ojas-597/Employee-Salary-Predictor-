import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("adult_3.csv")

print(df.columns)

# -----------------------------
# Select only required columns
# -----------------------------
required_cols = [
    "age",
    "gender",
    "education",
    "occupation",
    "hours-per-week",
    "experience",
    "salary"
]

# Keep only required columns
data = data[required_cols]

# -----------------------------
# Encode categorical columns
# -----------------------------
for col in ["gender", "education", "occupation"]:
    data[col] = data[col].astype("category").cat.codes

# Encode target column
data["salary"] = data["salary"].astype("category").cat.codes

# -----------------------------
# Split features and target
# -----------------------------
X = data.drop("salary", axis=1)
y = data["salary"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "best_model.pkl")

print("✅ Model trained and saved as best_model.pkl")
