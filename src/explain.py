import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/attrition_rf.pkl")

# Load data
df = pd.read_csv("data/ibm_hr.csv")

# Drop unused columns
drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]
df.drop(columns=drop_cols, inplace=True)

# Encode target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Features only
X = df.drop("Attrition", axis=1)

# Encode categorical columns
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category").cat.codes

# --- SHAP EXPLAINABILITY (SAFE METHOD) ---

# Use a small sample for speed
X_sample = X.sample(300, random_state=42)

# Create explainer using predict_proba
explainer = shap.Explainer(model.predict_proba, X_sample)

# Calculate SHAP values
shap_values = explainer(X_sample)

# Global feature importance (attrition class = 1)
shap.summary_plot(
    shap_values[:, :, 1],
    X_sample,
    plot_type="bar"
)
plt.show()

# Detailed impact plot
shap.summary_plot(
    shap_values[:, :, 1],
    X_sample
)
plt.show()
