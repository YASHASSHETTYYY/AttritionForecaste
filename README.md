# AttritionForecaste

A practical machine learning project for predicting employee attrition and helping HR teams prioritize retention strategies.

## Project Overview

**AttritionForecaste** predicts the likelihood that an employee may leave the organization, using the IBM HR analytics dataset and a Random Forest classifier.

The project includes:
- Data exploration scripts for understanding attrition patterns.
- Preprocessing and class-imbalance handling with SMOTE.
- Model training and evaluation workflow.
- SHAP-based explainability for transparent model behavior.
- A Streamlit web app for batch scoring and ROI simulation.

This repository is designed for both:
- **Beginners** who want a complete, runnable ML project structure.
- **Experienced users** who want to iterate quickly on modeling and deployment ideas.

---

## Features

- ✅ End-to-end attrition prediction pipeline.
- ✅ Automatic preprocessing and categorical encoding.
- ✅ Class imbalance mitigation using `SMOTE`.
- ✅ Model persistence with `joblib`.
- ✅ Explainability using SHAP summary plots.
- ✅ Interactive Streamlit app with:
  - CSV upload
  - Attrition probability scoring
  - High-risk employee identification
  - Retention ROI simulation

---

## Project Structure

```text
AttritionForecaste/
├── app/
│   └── app.py                   # Streamlit application
├── data/
│   └── ibm_hr.csv               # Source dataset
├── docs/
│   └── explainability_insights.md
├── models/
│   └── attrition_rf.pkl         # Trained Random Forest model artifact
├── src/
│   ├── eda.py                   # Exploratory data analysis
│   ├── preprocessing.py         # Data preprocessing + SMOTE prep check
│   ├── train.py                 # Training and evaluation script
│   └── explain.py               # SHAP explainability script
├── requirements.txt
└── README.md
```

---

## Installation

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd AttritionForecaste
```

### 2) Create and activate a virtual environment (recommended)

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

Main dependencies used in this project:
- `pandas`, `numpy` – data handling
- `scikit-learn` – model training/evaluation
- `imbalanced-learn` – SMOTE oversampling
- `matplotlib`, `seaborn` – visualization
- `streamlit` – web UI
- `shap` – model explainability
- `joblib` – model serialization

Additional packages currently listed:
- `mlflow`, `fastapi`, `uvicorn` (available for future extension; not required by current scripts)

---

## Usage

Run the following from the repository root.

### 1) Exploratory Data Analysis

```bash
python src/eda.py
```

What this does:
- Prints shape, columns, missing values, and class distribution.
- Opens plots for attrition distribution and selected feature relationships.

### 2) Preprocessing Sanity Check

```bash
python src/preprocessing.py
```

What this does:
- Applies the same preprocessing strategy used by the model.
- Shows class counts before and after SMOTE.

### 3) Train the Model

```bash
python src/train.py
```

What this does:
- Loads and preprocesses data.
- Trains a Random Forest classifier.
- Prints accuracy, confusion matrix, and classification report.
- Saves model to:
  - `models/attrition_rf.pkl`

### 4) Generate Explainability Plots

```bash
python src/explain.py
```

What this does:
- Loads the trained model.
- Computes SHAP values on a sample.
- Displays global importance and detailed impact plots.

### 5) Run the Streamlit App

```bash
streamlit run app/app.py
```

Then open the URL shown by Streamlit (usually `http://localhost:8501`).

---

## Configuration Options

This project currently uses script-level constants (hardcoded defaults). You can customize behavior by editing scripts directly.

### Training (`src/train.py`)

- Train/test split:
  - `test_size=0.2`
  - `random_state=42`
  - `stratify=y`
- SMOTE:
  - `random_state=42`
- Random Forest:
  - `n_estimators=200`
  - `max_depth=10`
  - `n_jobs=-1`

### Explainability (`src/explain.py`)

- SHAP sample size:
  - `X.sample(300, random_state=42)`

### Streamlit App (`app/app.py`)

- High-risk threshold:
  - `Attrition_Probability > 0.7`
- ROI simulation defaults:
  - Average replacement cost: `₹1,000,000`
  - Retention success slider: default `20%`

> Tip: For production usage, consider moving these values into environment variables or a dedicated config file.

---

## Input Data Expectations

To score employee data in the Streamlit app:
- Upload a CSV with feature columns consistent with the training data schema.
- Optional: include `Attrition` column; the app will ignore it during prediction.
- Columns like `EmployeeCount`, `EmployeeNumber`, `Over18`, and `StandardHours` are dropped automatically.

If input columns differ significantly from the training schema, predictions may fail or become unreliable.

---

## Troubleshooting

### 1) `FileNotFoundError` for dataset or model

- Confirm you are running commands from the repository root.
- Ensure these files exist:
  - `data/ibm_hr.csv`
  - `models/attrition_rf.pkl` (generated after training)

### 2) SHAP plotting issues or slow execution

- SHAP can be computationally heavy.
- Reduce sample size in `src/explain.py` (e.g., from 300 to 100).
- Ensure your Python environment has compatible versions of `numpy`, `shap`, and `matplotlib`.

### 3) Streamlit app fails at startup

- Verify dependencies are installed from `requirements.txt`.
- Retrain the model if `models/attrition_rf.pkl` is missing.
- Restart app:
  ```bash
  streamlit run app/app.py
  ```

### 4) Unexpected model behavior

- Re-check categorical encoding consistency between training and inference.
- Use SHAP plots (`src/explain.py`) to inspect feature influence.
- Confirm the uploaded CSV matches expected column names and data types.

---

## Contribution Guidelines

Contributions are welcome! To keep collaboration smooth:

1. **Fork** the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-change
   ```
3. Make focused, well-documented changes.
4. Run local checks before submitting:
   - training script executes
   - app launches
   - no broken imports or syntax errors
5. Commit with a clear message:
   ```bash
   git commit -m "Add <short description>"
   ```
6. Open a Pull Request with:
   - What changed
   - Why it changed
   - How to test

### Suggested contribution areas
- Better feature engineering and encoding strategy.
- Hyperparameter tuning and model comparison.
- Config-driven pipeline (YAML/ENV-based settings).
- Improved model validation and calibration.
- Containerization and production deployment pipeline.

---

## Roadmap Ideas

- Add unit tests and CI pipeline.
- Add experiment tracking with MLflow.
- Add API serving layer (FastAPI).
- Add robust preprocessing pipeline object (`sklearn.Pipeline`) to avoid train/inference drift.
- Add model performance dashboard and fairness checks.

---



---

## Acknowledgments

- IBM HR Analytics Employee Attrition dataset (source used in this repository).
- Open-source Python ecosystem (`scikit-learn`, `streamlit`, `shap`, and others) powering the workflow.

