'''
Sinh viên thực hiện: Nguyễn Huỳnh Hải Đăng
MSSV: 20521159
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml
import sklearn.datasets as datasets
import math

#1. Markdown
st.markdown("""
# Hệ thống AI dự báo
""")
#data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="data")
setting={}
class count_section:
    def __init__(self):
        self.number = 0
    def __call__(self):
        self.number+=1
        return self.number

if True:
    get_num_section = count_section()

    # Select input feature
    st.markdown(f"""## {get_num_section()}.Data""")

    X,y = datasets.load_wine(return_X_y=True, as_frame=True)
    df = X.copy()
    df["class"]=y
    df = df.sample(frac=1).reset_index(drop=True)

    dataset = ml.Dataset(df,"df")

    st.dataframe(dataset.origin_data)

    if "select_all_status" not in st.session_state:
        st.session_state["select_all_status"] = False

    def set_select_all_status():
        st.session_state.select_all_status = True
    def reset_select_all_status():
        st.session_state.select_all_status = False

    get_feature_inputs={}
    target = dataset.origin_data.columns[-1]

    st.markdown(f"""## {get_num_section()}.Select features as input""")
    st.write("Select features as input features")
    if not st.session_state.select_all_status:
        for i in list(dataset.origin_data.columns)[:-1]:
            get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}")})
    else:
        get_feature_inputs={}
        for i in list(dataset.origin_data.columns)[:-1]:
            get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}",value=True)})
    st.button("Select all", on_click = set_select_all_status)
    st.button("Clear all", on_click = reset_select_all_status)
    n_selected_features = np.sum(list(get_feature_inputs.values()))

    # Select output feature
    st.write(f"Output features: {target}")
    if "k_fold_select" not in st.session_state:
        st.session_state["k_fold_select"]=False
    def set_kflod():
        st.session_state["k_fold_select"]=not st.session_state["k_fold_select"]
        
    if not st.session_state["k_fold_select"]:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Select K-Fold
            st.markdown(f"""## {get_num_section()}.Select K-Fold Cross validation""")
            kfold = st.checkbox(f"K-Fold Cross validation",key="kfold",on_change =set_kflod)
            #st.session_state.k_fold_select=kfold
            setting.update({"kfold":kfold})

            if kfold:
                if n_selected_features<3:
                    st.error("Please select at least THREE features")
                else:
                    K = st.slider(f"Select K between 2 and {n_selected_features}",key="rate_train", step=1, min_value=2, max_value=int(n_selected_features), value=2)
                    setting.update({"K":K})
            # Select the rate of train dataset
        with col2:
            if kfold == False:
                st.markdown(f"""## {get_num_section()}.Select Train/test split""")
                rate = st.slider("Enter the rate of train dataset",key="rate_train", value=0.1, min_value=0.1, max_value=1.0, step=0.01)
                st.write(f"Train rate : {int(rate*100)}%")
                st.write(f"Test rate : {int((1-rate)*100)}%")
                setting.update({"rate":rate})
        with col3:
            # Select PCA
            st.markdown(f"""## {get_num_section()}.Select PCA""")
            st.write(f"Do you want to use PCA for reducing demensions")
            pca = st.checkbox(f"PCA",key="PCA")
            if pca:
                if n_selected_features<2:
                    st.error("Please select at least TWO a feature")
                else:
                    reduce = st.slider(f"Select components bewteen 1 and {n_selected_features}",key="pca_n", step=1, min_value=1, max_value = int(n_selected_features), value=1)
                    setting.update({"pca_n":reduce})
            setting.update({"pca":pca})
        with col4:
            # Select metric
            st.markdown(f"""## {get_num_section()}.Select metrics""")
            st.write("Select at least one metric for evaluating model")
            setting.update({"F1":st.checkbox(f"F1",key="F1")})
            setting.update({"LogLoss":st.checkbox(f"LogLoss",key="LogLoss")})
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            # Select K-Fold
            st.markdown(f"""## {get_num_section()}.Select K-Fold Cross validation""")
            kfold = st.checkbox(f"K-Fold Cross validation",key="kfold",on_change =set_kflod)
            setting.update({"kfold":kfold})

            if kfold:
                if n_selected_features<3:
                    st.error("Please select at least THREE features")
                else:
                    K = st.slider(f"Select K between 2 and {n_selected_features}",key="rate_train", step=1, min_value=2, max_value=int(n_selected_features), value=2)
                    setting.update({"K":K})
            # Select the rate of train dataset
        with col2:
            # Select PCA
            st.markdown(f"""## {get_num_section()}.Select PCA""")
            st.write(f"Do you want to use PCA for reducing demensions")
            pca = st.checkbox(f"PCA",key="PCA")
            if pca:
                if n_selected_features<2:
                    st.error("Please select at least TWO a feature")
                else:
                    reduce = st.slider(f"Select components bewteen 1 and {n_selected_features}",key="pca_n", step=1, min_value=1, max_value = int(n_selected_features), value=1)
                    setting.update({"pca_n":reduce})
            setting.update({"pca":pca})
        with col3:
            # Select metric
            st.markdown(f"""## {get_num_section()}.Select metrics""")
            st.write("Select at least one metric for evaluating model")
            setting.update({"F1":st.checkbox(f"F1",key="F1")})
            setting.update({"LogLoss":st.checkbox(f"LogLoss",key="LogLoss")})
    
    if st.button("Run",key="run"):
        if n_selected_features==0:
            st.error("Please select at lease a feature")
        elif (setting["F1"] or setting["LogLoss"])==False:
            st.error("Please select a metric for evaluation")
        else:
            feature_selected = [ i for i in get_feature_inputs.keys() if get_feature_inputs[i]]
            if len(feature_selected)<1:
                st.error("Please select at least a feature")
            else:
                setting.update({"feature_list":feature_selected,"target":target})
                model = ml.Model_AI(dataset, setting)
                model.fit()

                stats = model.get_value_metrics
                for i in stats.keys():
                    st.write(f"{i} : {stats[i]}")
                st.pyplot(model.plot_history())