# Telco Customer Churn — ML Assignment 2 (BITS WILP)

**Author:** MADHURENDRA KUMAR  
**Course:** Machine Learning — Assignment 2  
**Due:** 15-Feb-2026 23:59 IST  citeturn1search1

> End‑to‑end ML workflow for a **classification** problem: data prep, model training & evaluation (6 models), and a **Streamlit** app for interactive demo & deployment. citeturn1search1

---

## 1) Problem Statement
Predict whether a telecom customer will **churn** (leave the service), using the public *Telco Customer Churn* dataset. The task is a supervised **binary classification** problem where the target variable `Churn` is mapped to 1 (Yes) and 0 (No). The solution includes model training, evaluation on multiple metrics, and a lightweight UI for inference. citeturn1search2

---

## 2) Dataset Description
- **Dataset file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
- **Shape after cleaning:** **7043 rows × 20 columns**  
- **Target:** `Churn` (Yes/No → 1/0)  
- **Key features (subset):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.  
- **Cleaning highlights:** Dropped `customerID` if present; coerced `TotalCharges` to numeric and imputed missing values with the **median**; normalized/coded features through a `ColumnTransformer` (StandardScaler for numerics and One‑Hot for categoricals). citeturn1search2

> **Note for evaluators:** The dataset meets the **minimum instance (≥500)** and **feature (≥12)** requirements. citeturn1search1

---

## 3) Models & Metrics
We trained **six** classifiers on the same dataset and reported the following metrics on the hold‑out test set: **Accuracy, AUC, Precision, Recall, F1, MCC**. citeturn1search1

**Models implemented**
1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble; fallback to Gradient Boosting if XGBoost not available)  citeturn1search1turn1search2

### 3.1 Comparison Table (Test Set)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (Ensemble) | 0.8062 | 0.8432 | 0.6735 | 0.5241 | 0.5895 | 0.4715 citeturn1search2 |
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 citeturn1search2 |
| Random Forest (Ensemble) | 0.7779 | 0.8173 | 0.6048 | 0.4706 | 0.5293 | 0.3921 citeturn1search2 |
| Naive Bayes (Gaussian) | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 citeturn1search2 |
| kNN | 0.7644 | 0.8047 | 0.5544 | 0.5722 | 0.5632 | 0.4020 citeturn1search2 |
| Decision Tree | 0.7289 | 0.6573 | 0.4896 | 0.5053 | 0.4974 | 0.3119 citeturn1search2 |

> The confusion matrix for the **best AUC** model is plotted in the notebook; all trained model pipelines and metrics JSON are saved under `./models/`. citeturn1search2

### 3.2 Observations
- **Logistic Regression vs. XGBoost:** Both achieve similar **AUC ≈ 0.84**, indicating strong ranking performance; **LR** has the highest **F1 (0.604)**, suggesting a slightly better balance of precision/recall on this split. citeturn1search2
- **Naive Bayes (Gaussian):** Exhibits **high recall (0.8369)** but **low precision (0.4589)**, consistent with many false positives—useful when missing a churner is costlier than flagging a non-churner. citeturn1search2
- **Random Forest & kNN:** Mid‑tier performance with **AUC ≈ 0.80–0.82**; better precision than NB but lower recall than LR/NB. citeturn1search2
- **Decision Tree:** Lowest **AUC (0.6573)**, likely underfitting/overfitting without additional tuning or ensembling. citeturn1search2

---

## 4) Repository Structure (as required)
```
project-folder/
├── app.py                  # Streamlit app entry point
├── requirements.txt        # Python dependencies for deployment
├── README.md               # You are here
└── model/                  # Saved model files (e.g., .joblib) or notebooks (.ipynb)
```
> Keep this structure in your GitHub repo to avoid deployment issues. citeturn1search1

---

## 5) Environment & Requirements
Create a virtual environment and install the following (minimum set per assignment + libs used in notebook):
```bash
pip install -U streamlit scikit-learn numpy pandas matplotlib seaborn joblib xgboost
```
> Missing dependencies are the #1 cause of deployment failure—ensure `requirements.txt` contains these packages. citeturn1search1

Example `requirements.txt`:
```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
xgboost
```

---

## 6) How to Run (Local)
1. **Clone** the repo and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root. citeturn1search2  
2. (Optional) **Reproduce notebook**: open the Jupyter notebook to train/evaluate and export models/metrics to `./models/`. citeturn1search2  
3. **Run Streamlit** app locally:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL shown by Streamlit to interact with the UI.

---

## 7) Streamlit App: Required Features
Your app must include **at least**:  
- **CSV upload** (upload **test** data only, per free-tier limits).  
- **Model selection** dropdown (for multiple models).  
- Display of core **evaluation metrics**.  
- **Confusion matrix** or **classification report** visualization.  
And it must be **deployed on Streamlit Community Cloud** with a working link. citeturn1search1

### 7.1 Deployment (Streamlit Community Cloud)
1. Go to <https://streamlit.io/cloud> and sign in with GitHub.  
2. Click **New app** → Select your repo → Choose branch (e.g., `main`) → pick `app.py` → **Deploy**.  
3. Wait a few minutes; the app should go live and be shareable via a public URL. citeturn1search1

**Placeholders (replace after deployment):**  
- **GitHub Repo:** *add your public link here*  
- **Live Streamlit App:** *add your deployed link here*  

---

## 8) BITS Virtual Lab Proof
Perform the assignment on **BITS Virtual Lab** and include **one screenshot** of execution as proof in your submission PDF. citeturn1search1

---

## 9) Anti‑Plagiarism & Academic Integrity (Summary)
Submissions are checked at code, UI, and model levels. Identical repo structures/variable names, uncustomized templates, or same dataset+model+outputs across students may be flagged. Only use AI tools for learning support; **no direct copy‑paste**. citeturn1search1

**Final Checklist before submission:**  
- ✅ GitHub repo link works  
- ✅ Streamlit app link opens and loads without errors  
- ✅ All required app features implemented  
- ✅ `README.md` contents included in the submitted PDF  
- ✅ Screenshot from BITS Lab added  citeturn1search1

---

## 10) Reproducibility Notes (from Notebook)
- **Preprocessing:** `ColumnTransformer` with `StandardScaler` (numerics) + `OneHotEncoder(handle_unknown='ignore')` (categoricals).  
- **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)`.  
- **Pipelines:** `Pipeline([('prep', preprocessor), ('model', MODEL)])`.  
- **Saved artifacts:** `./models/*.joblib`, `./models/metrics.json`, `./models/reports.json`.  
- **Visualization:** Confusion matrix for best‑AUC model.  
These are implemented in the provided Jupyter notebook. citeturn1search2

---

## 11) License & Acknowledgments
- This project is for academic evaluation under **BITS WILP — Machine Learning Assignment 2**.  
- Dataset credit: *Telco Customer Churn* (public educational dataset).  
- Built with: Python, scikit‑learn, Streamlit, (optional) XGBoost.  

