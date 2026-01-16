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

     st.markdown("--------------Default File WA_Fn-UseC-Telco-Customer-Churn.csv---------------------------")    

     csv_url = "https://github.com/madhurendrabtech/mlassignment2/blob/main/WA_Fn-UseC-Telco-Customer-Churn.csv"

     df = pd.read_csv(csv_url)

     st.dataframe(df.head())
 
st.markdown("--------------This is reports for all model selection of WA_Fn-UseC-Telco-Customer Churn.csv---------------------------")
 
# Fetch and display metrics

url = "https://github.com/madhurendrabtech/mlassignment2/blob/main/Models/reports.json"  # 

if url:

    response = requests.get(url)

    if response.status_code == 200:

        metrics_text = response.text

        st.subheader(f"Evaluation Metrics for {model_option}")

        st.markdown(f"```\n{metrics_text}\n```")  # preserves formatting

    else:

        st.error(f"Could not fetch the metrics file for {model_option}")

else:

    st.warning("No metrics available for the  models")

 
st.markdown("--------------This is Matrix for all model selection of WA_Fn-UseC-Telco-Customer Churn.csv---------------------------")
 
# Fetch and display metrics

url = "https://github.com/madhurendrabtech/mlassignment2/blob/main/Models/metrics.json"  # 

if url:

    response = requests.get(url)

    if response.status_code == 200:

        metrics_text = response.text

        st.subheader(f"Evaluation Metrics for {model_option}")

        st.markdown(f"```\n{metrics_text}\n```")  # preserves formatting

    else:

        st.error(f"Could not fetch the metrics file for {model_option}")

else:

    st.warning("No metrics available for the  models")

 
