import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/ibm_hr.csv")

# Drop useless columns (constant / IDs)
drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]
df.drop(columns=drop_cols, inplace=True)

# Encode target variable
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Separate features & target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Encode categorical features
cat_cols = X.select_dtypes(include="object").columns

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())
print("Train shape:", X_train_res.shape)
print("Test shape:", X_test.shape)
