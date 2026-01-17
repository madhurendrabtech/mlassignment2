
import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import json

# -----------------------------
# Page Config & Header
# -----------------------------
st.set_page_config(page_title="Bits ML Classification – Models & Metrics", layout="wide")
st.title("Classification Models and Evaluation Metrics")
st.markdown("""
This app trains six classification models on a dataset and reports the following metrics:
- **Accuracy**
- **AUC Score**
- **Precision**
- **Recall**
- **F1 Score**
- **Matthews Correlation Coefficient (MCC)**

You can upload your own dataset (CSV; ≥ 12 features and ≥ 1000 rows), or use the default.
""")

# -----------------------------
# Sidebar - Data selection
# -----------------------------
with st.sidebar:
    st.header("Data & Settings")
    data_choice = st.selectbox("Choose dataset", ["Default (Telco Churn)", "Upload CSV"])
    # Uncomment if you want model training options:
    # test_size = st.slider("Test Size (Validation Split)", 0.1, 0.4, 0.2, 0.02)
    # scale_numeric = st.checkbox("Scale numeric features (StandardScaler)", value=True)
    # model_option = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "SVM", "XGBoost"])
    # random_state = st.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def fetch_json(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def safe_json_loads(x):
    """Handles cases where a DataFrame cell contains a JSON string."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def classification_report_to_df(report_dict: dict) -> pd.DataFrame:
    """
    Convert sklearn classification_report(output_dict=True) into a tidy DataFrame.
    Handles 'accuracy' key specially.
    """
    rows = []
    for label, metrics in report_dict.items():
        if label == "accuracy":
            rows.append({
                "class": "accuracy",
                "precision": np.nan,
                "recall": np.nan,
                "f1-score": metrics,
                "support": np.nan
            })
        elif isinstance(metrics, dict):
            row = {"class": label}
            row.update(metrics)
            rows.append(row)
    df_rep = pd.DataFrame(rows)
    # Order columns
    cols = ["class", "precision", "recall", "f1-score", "support"]
    df_rep = df_rep[[c for c in cols if c in df_rep.columns]].copy()
    # Move accuracy on top if present
    if "class" in df_rep.columns:
        df_rep["__order__"] = df_rep["class"].apply(lambda x: 0 if x == "accuracy" else 1)
        df_rep = df_rep.sort_values(["__order__", "class"]).drop(columns="__order__")
    # Round numeric
    for c in ["precision", "recall", "f1-score"]:
        if c in df_rep.columns:
            df_rep[c] = pd.to_numeric(df_rep[c], errors="coerce").round(4)
    if "support" in df_rep.columns:
        df_rep["support"] = pd.to_numeric(df_rep["support"], errors="coerce").astype("Int64")
    return df_rep

def render_reports_json(data):
    """
    Renders the Models/reports.json content.
    We try to support:
      - list[dict] where each dict is a model with a 'report' field (string/dict)
      - dict keyed by model names
      - flat dict having a single report
    """
    st.subheader("Reports (Per-Model Classification Reports)")
    # Case A: list of models
    if isinstance(data, list):
        # Normalize each entry
        tabs = st.tabs([f"Model {i+1}" for i in range(len(data))])
        for i, entry in enumerate(data):
            with tabs[i]:
                st.write("Raw entry (normalized):")
                st.json(entry)
                # Try to find a report-like object
                candidate = None
                for key in ["report", "classification_report", "metrics", "results"]:
                    if key in entry:
                        candidate = entry[key]
                        break
                if candidate is None:
                    # maybe the entry itself is the report dict
                    candidate = entry
                candidate = safe_json_loads(candidate)
                if isinstance(candidate, dict):
                    df_rep = classification_report_to_df(candidate)
                    st.dataframe(df_rep, use_container_width=True)
                else:
                    st.info("Could not find a classification report dictionary in this entry.")
    # Case B: dict keyed by model names
    elif isinstance(data, dict):
        # If it looks like a single report (has 'accuracy' and class keys), render directly
        looks_like_report = any(k in data for k in ["accuracy", "macro avg", "weighted avg", "0", "1"])
        if looks_like_report and all(isinstance(v, (dict, float, int)) for v in data.values()):
            st.dataframe(classification_report_to_df(data), use_container_width=True)
        else:
            # Treat as dict-of-models
            tabs = st.tabs(list(data.keys()))
            for (model_name, value), tab in zip(data.items(), tabs):
                with tab:
                    st.caption(f"**{model_name}**")
                    st.json(value)
                    value = safe_json_loads(value)
                    if isinstance(value, dict):
                        looks_like_report = any(k in value for k in ["accuracy", "macro avg", "weighted avg", "0", "1"])
                        if looks_like_report:
                            st.dataframe(classification_report_to_df(value), use_container_width=True)
                        else:
                            # Try deep search for a report-like subkey
                            found = None
                            for key in ["report", "classification_report", "metrics", "results"]:
                                if key in value and isinstance(value[key], (dict, str)):
                                    found = safe_json_loads(value[key])
                                    break
                            if isinstance(found, dict):
                                st.dataframe(classification_report_to_df(found), use_container_width=True)
                            else:
                                st.info("No classification report found for this model.")
                    else:
                        st.info("Model value is not a dict; showing as JSON above.")
    else:
        st.info("Unsupported JSON root type for reports.json. Showing raw JSON below.")
        st.json(data)

def render_metrics_json(data):
    """
    Renders Models/metrics.json as a clean table.
    Supports list of dicts or dict-of-dicts.
    """
    st.subheader("Aggregate Metrics (All Models)")
    try:
        if isinstance(data, list):
            dfm = pd.DataFrame(data)
        elif isinstance(data, dict):
            # dict-of-dicts → rows are models
            dfm = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "model"})
        else:
            # attempt to normalize anything else
            dfm = pd.json_normalize(data)
        # Order useful columns if present
        preferred = ["model", "accuracy", "auc", "precision", "recall", "f1", "mcc"]
        cols = [c for c in preferred if c in dfm.columns] + [c for c in dfm.columns if c not in preferred]
        dfm = dfm[cols]
        # Round metrics
        for c in ["accuracy", "auc", "precision", "recall", "f1", "mcc"]:
            if c in dfm.columns:
                dfm[c] = pd.to_numeric(dfm[c], errors="coerce").round(4)
        st.dataframe(dfm, use_container_width=True)
        return dfm
    except Exception as e:
        st.error(f"Could not render metrics.json as table: {e}")
        with st.expander("Raw metrics.json"):
            st.json(data)
        return None

# -----------------------------
# Dataset load
# -----------------------------
df = None
if data_choice == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file (last column as target is recommended)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Uploaded: {uploaded.name}")
else:
    st.markdown("**Using default dataset:** `WA_Fn-UseC-Telco-Customer-Churn.csv`")
    csv_url = "https://raw.githubusercontent.com/madhurendrabtech/mlassignment2/main/WA_Fn-UseC-Telco-Customer-Churn.csv"
    try:
        df = fetch_csv(csv_url)
    except Exception as e:
        st.error(f"Failed to fetch default CSV: {e}")

if df is not None:
    st.write("**Preview:**")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.divider()
st.markdown("### Reports for models on `WA_Fn-UseC-Telco-Customer-Churn.csv`")

# -----------------------------
# Reports.json
# -----------------------------
REPORTS_URL = "https://raw.githubusercontent.com/madhurendrabtech/mlassignment2/main/Models/reports.json"
try:
    reports_data = fetch_json(REPORTS_URL)
    render_reports_json(reports_data)
    with st.expander("Raw reports.json"):
        st.json(reports_data)
except requests.HTTPError as e:
    st.error(f"HTTP error while fetching reports.json: {e}")
except requests.RequestException as e:
    st.error(f"Network error while fetching reports.json: {e}")
except ValueError as e:
    st.error(f"Failed to parse reports.json: {e}")

st.divider()
st.markdown("### Metrics for all model selections of `WA_Fn-UseC-Telco-Customer-Churn.csv`")

# -----------------------------
# Metrics.json
# -----------------------------
METRICS_URL = "https://raw.githubusercontent.com/madhurendrabtech/mlassignment2/main/Models/metrics.json"
try:
    metrics_data = fetch_json(METRICS_URL)
    df_metrics = render_metrics_json(metrics_data)
    with st.expander("Raw metrics.json"):
        st.json(metrics_data)

    # Optional: small chart to compare F1 if available
    if isinstance(df_metrics, pd.DataFrame) and "model" in df_metrics.columns:
        numeric_f1 = pd.to_numeric(df_metrics.get("f1", pd.Series(dtype=float)), errors="coerce")
        if numeric_f1.notna().any():
            st.markdown("#### F1 Score by Model")
            st.bar_chart(df_metrics.set_index("model")["f1"])
except requests.HTTPError as e:
    st.error(f"HTTP error while fetching metrics.json: {e}")
except requests.RequestException as e:
    st.error(f"Network error while fetching metrics.json: {e}")
except ValueError as e:
    st.error(f"Failed to parse metrics.json: {e}")

