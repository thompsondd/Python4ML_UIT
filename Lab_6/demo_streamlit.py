import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

#st.title("My First Dang's Web")

#1. Markdown
st.markdown("""
# Hướng dẫn streamlit
## 1. Giới thiệu
## 2. Các phép toán cơ bản
""")
#2. Button
#3. Combo box/ Drop box
#4. Radio box
#5. Check box
def function():
    a = st.text_input("Nhập a")
    b = st.text_input("Nhập b")
    operators = st.selectbox("Chọn các phép toán",["Cộng","Trừ","Nhân","Chia"])
    if st.button("Tính"):
        a = int(a)
        b = int(b)
        if operators == "Cộng":
            st.text_input("Kết quả: ",a+b)
        elif operators == "Trừ":
            st.text_input("Kết quả: ",a-b)
        elif operators == "Nhân":
            st.text_input("Kết quả: ",a*b)
        elif operators == "Chia":
            st.text_input("Kết quả: ",a*1.0/b)
#6. Group/Tab
tab1, tab2, tab3 = st.tabs(["Calculator", "Dog", "Owl"])

with tab1:
   st.header("Calculator")
   function()
   #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
#7. Upload file
def image_process(img):
    filter1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter2 = np.array([[-1,2,1],[0,0,0],[-1,-2,1]])
    Ix = cv.filter2D(img,-1, filter1.T)
    Iy = cv.filter2D(img,-1, filter2.T)
    return cv.add(Ix,Iy)
    
data = st.file_uploader("Đưa ảnh đây để còn hiện")
if data is not None:
    byte_data=data.getvalue()
    with open(f"./data/{data.name}","wb") as f:
        f.write(byte_data)
    
    col_in, col_out = st.columns(2)
    with col_in:
        st.header("Ảnh input")
        st.image(Image.open(f"./data/{data.name}"))
    with col_out:
        st.header("Ảnh input")
        st.image(image_process(cv.imread(f"./data/{data.name}",0)))