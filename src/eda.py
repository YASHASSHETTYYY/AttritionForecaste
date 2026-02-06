import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/ibm_hr.csv")

# Basic inspection
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nAttrition distribution:\n", df["Attrition"].value_counts())

# Convert Attrition to numeric for plots
df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Plot attrition count
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.show()

# Attrition vs Job Satisfaction
sns.boxplot(x="Attrition", y="JobSatisfaction", data=df)
plt.title("Job Satisfaction vs Attrition")
plt.show()

# Attrition vs Monthly Income
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")
plt.show()
