import streamlit as st
import os
import requests
import subprocess
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bits ML Classification and Models & Metrics")
st.title("Classification Model and Evaluation matrix")
st.markdown("""
This app train six classification model on a single data set and reports the following  metrices:
- Accurecy
- AUC Score
- Precesion
- Recall
- F1 Score
- Mathhews corelation coefficent (MCC)

 you can upload your owndataset(CSV) min 12 features and 1000 instances).           
""")
with st.sidebar:
     st.header("Data & Setting")
     data_choice=st.selectbox("choose dataset",["Upload CSV","Uploaad Excel CSV supported"])
     #test_size=st.slider("Test Size (Validation Split)",0.1,0.4,0.2,0.02)
     #scale_numeric=st.checkbox("Scale Numeric features(StandradScalar)",value=True)
     model_option = st.selectbox("Select Model",["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "SVM", "XGBoost"])
     #random_state=st.number_input("Random seed",min_value=0,max_value=10000, value=42,step=1)
if data_choice=="Upload CSV":
     uploaded=st.file_uploader("Upload the CSV file (last colum target recomended)",type=["csv"])
     df=None
     if uploaded is not None:
        df=pd.read_csv(uploaded)
        st.write("preview:",df.head())
        st.write("shape",df.shape)
  
      
else:
     st.markdown("--------------Default File Heart.csv---------------------------")    
     csv_url = "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/data/heart.csv"
     df = pd.read_csv(csv_url)
     st.dataframe(df.head())
 
st.markdown("--------------Matrix for all 6 Model of Heart.csv---------------------------") 
csv_url="https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/metrics.csv"
df = pd.read_csv(csv_url)
st.dataframe(df.head())
st.markdown("--------------This is Matrix for all model selection of Heart.csv---------------------------") 

metrics_urls = {
    "Logistic Regression": "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/LogisticRegression_report.txt",
    "Random Forest":       "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/RandomForest_report.txt",
    "KNN":                 "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/kNN_report.txt",
    "Decision Tree":       "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/Decision_Tree_report.txt",
    "SVM":                 "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/NaiveBayes(Gaussian)_report.txt",
    "XGBoost":             "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/XGBoost_report.txt",
}

# Fetch and display metrics
url = metrics_urls.get(model_option)  # Get URL for selected model
if url:
    response = requests.get(url)
    if response.status_code == 200:
        metrics_text = response.text
        st.subheader(f"Evaluation Metrics for {model_option}")
        st.markdown(f"```\n{metrics_text}\n```")  # preserves formatting
    else:
        st.error(f"Could not fetch the metrics file for {model_option}")
else:
    st.warning("No metrics available for the selected model")
         
   
