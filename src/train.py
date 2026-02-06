print("=== TRAINING SCRIPT STARTED ===")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("data/ibm_hr.csv")

# Drop useless columns
drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]
df.drop(columns=drop_cols, inplace=True)

# Encode target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Split features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Encode categorical columns
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category").cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("=== DATA PREPARED ===")

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)
print("=== MODEL TRAINED ===")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/attrition_rf.pkl")
print("\n=== MODEL SAVED TO models/attrition_rf.pkl ===")

print("=== TRAINING SCRIPT FINISHED ===")
