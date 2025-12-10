import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("adult 3.csv")

# Print column names (for checking)
print("✅ Columns found:", data.columns)

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

# Check if columns exist
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise Exception(f"❌ Missing columns in CSV: {missing_cols}")

# Keep only required columns
data = data[required_cols]

# -----------------------------
# Encode categorical columns
# -----------------------------
for col in ["gender", "education", "occupation"]:
    data[col] = data[col].astype("category").cat.codes

# Encode target column
data["income"] = data["income"].astype("category").cat.codes

# -----------------------------
# Split features and target
# -----------------------------
X = data.drop("income", axis=1)
y = data["income"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "best_model.pkl")

print("✅ Model trained and saved as best_model.pkl")
