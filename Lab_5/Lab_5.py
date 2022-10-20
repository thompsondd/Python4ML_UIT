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
data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="salary")
if data is not None:
    byte_data=data.getvalue()
    with open(f"./data/{data.name}","wb+") as f:
        f.write(byte_data)
    model_salary = ml.Model_AI()
    model_salary.fit(f"./data/{data.name}")
    get_inputs={}
    for i in list(model_salary.origin_data.columns):
        if i !=model_salary.target:
            if i not in model_salary.str_col:
                get_inputs.update({i:st.text_input(f"Nhap {i}:",key=f"{i}")})
            else:
                get_inputs.update({i:st.selectbox(f"Chon {i}",model_salary.origin_data[i].unique(),key=f"{i}")})
    if st.button("Dự đoán lương",key="salary"):
        for i in get_inputs:
            if i not in model_salary.str_col:
                get_inputs[i]=float(get_inputs[i])
        #st.write(f"Input={get_inputs}")
        #st.write(f"Best model={model.best_model}")
        st.success(f"Salary = {model_salary.predict(get_inputs)}")
#function()
st.markdown("""## 2. Dự báo lợi nhuận""")
data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="profit")
if data is not None:
    byte_data=data.getvalue()
    with open(f"./data/{data.name}","wb+") as f:
        f.write(byte_data)
    model = ml.Model_AI()
    model.fit(f"./data/{data.name}")
    get_inputs={}
    for i in list(model.origin_data.columns):
        if i !=model.target:
            if i not in model.str_col:
                get_inputs.update({i:st.text_input(f"Nhap {i}:",key=f"{i}")})
            else:
                get_inputs.update({i:st.selectbox(f"Chon {i}",model.origin_data[i].unique(),key=f"{i}")})
    if st.button("Dự đoán lương",key="profit"):
        for i in get_inputs:
            if i not in model.str_col:
                get_inputs[i]=float(get_inputs[i])
        #st.write(f"Input={get_inputs}")
        #st.write(f"Best model={model.best_model}")
        st.success(f"Salary = {model.predict(get_inputs)}")
        #st.write(f"Salary = {model.predict(get_inputs)}")