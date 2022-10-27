import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import ml

#st.title("My First Dang's Web")

#1. Markdown
st.markdown("""
# Hệ thống AI dự báo
## 1. Dự báo lương
""")
data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="data")
if data is not None:
    byte_data=data.getvalue()
    with open(f"./data/{data.name}","wb+") as f:
        f.write(byte_data)
    # Select input feature
    dataset = ml.Dataset(f"./data/{data.name}")
    st.dataframe(dataset.origin_data)
    get_feature_inputs={}
    st.write("Select features as input features")
    for i in list(dataset.origin_data.columns):
        if i !=dataset.target:
            get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}")})
    # Select output feature
    st.write(f"Output features: {dataset.target}")
    # Select the rate of train dataset
    rate = st.number_input("Enter the rate of train dataset",key="rate_train", value=0.1, min_value=0.1, max_value=1.0, step=0.01)
    # Select K-Fold
    kfold = st.checkbox(f"K-Fold Cross validation",key="kfold")
    if kfold:
        K = st.number_input("Enter the rate of train dataset",key="rate_train", step=1)
    # Select metric
    st.write("Select at least one metric for evaluating model")
    mae = st.checkbox(f"MAE",key="MAE")
    mse = st.checkbox(f"MSE",key="MSE")
    

    
    if st.button("Run",key="run"):
        if (mae or mse)==False:
            st.error("Please select a metric for evaluation")
        else:
                model_salary = ml.Model_AI()
                model_salary.fit(f"./data/{data.name}")
        #for i in get_inputs:
        #    if i not in model_salary.str_col:
        #        get_inputs[i]=float(get_inputs[i])
        ##st.write(f"Input={get_inputs}")
        ##st.write(f"Best model={model.best_model}")
        #st.success(f"Salary = {model_salary.predict(get_inputs)}")