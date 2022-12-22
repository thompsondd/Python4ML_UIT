'''
Sinh viên thực hiện: Nguyễn Huỳnh Hải Đăng
MSSV: 20521159
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backend as ml
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
data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="data")
setting={}
if data is not None:
    columns1, columns2 = st.columns(2)
    byte_data=data.getvalue()
    with open(f"./data/{data.name}","wb+") as f:
        f.write(byte_data)

    get_num_section = count_section()

    dataset = ml.Dataset(f"./data/{data.name}")
    
    target = dataset.origin_data.columns[-1]
    st.subheader(f"Output feature: {target}")
    st.subheader("Data")
    st.dataframe(dataset.origin_data)

    with st.sidebar:
        if "select_all_status" not in st.session_state:
            st.session_state["select_all_status"] = False
        def set_select_all_status():
            st.session_state.select_all_status = True
        def reset_select_all_status():
            st.session_state.select_all_status = False

        get_feature_inputs={}
        

        #st.markdown(f"""## {get_num_section()}.Select features as input""")
        st.subheader("Select features as input features")
        if not st.session_state.select_all_status:
            for i in list(filter(lambda x: x!= target,list(dataset.origin_data.columns))):
                get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}",value=False)})
        else:
            get_feature_inputs={}
            for i in list(filter(lambda x: x!= target,list(dataset.origin_data.columns))):
                get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}",value=True)})
        
        colBut1, colBut2 =st.columns(2)
        with colBut1:
            st.button("Select all", on_click = set_select_all_status)
        with colBut2:
            st.button("Clear all", on_click = reset_select_all_status)
        n_selected_features = np.sum(list(get_feature_inputs.values()))

    # Select output feature
    if "k_fold_select" not in st.session_state:
        st.session_state["k_fold_select"]=False
    def set_kflod():
        st.session_state["k_fold_select"]=not st.session_state["k_fold_select"]
        
    if not st.session_state["k_fold_select"]:
        col1, col2, col3= st.columns(3)
        with col1:
            # Select K-Fold
            st.subheader(f"{get_num_section()}.Select K-Fold Cross validation")
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
                st.subheader(f"{get_num_section()}.Select Train/test split")
                rate = st.slider("Enter the rate of train dataset",key="rate_train", value=0.1, min_value=0.1, max_value=1.0, step=0.01)
                st.write(f"Train rate : {int(rate*100)}%")
                st.write(f"Test rate : {int((1-rate)*100)}%")
                setting.update({"rate":rate})
        with col3:
            # Select metric
            st.subheader(f"{get_num_section()}.Select metrics")
            st.write("Select at least one metric for evaluating model")
            setting.update({"F1":st.checkbox(f"F1",key="F1")})
            setting.update({"LogLoss":st.checkbox(f"LogLoss",key="LogLoss")})
    else:
        col1, col2 = st.columns(2)
        with col1:
            # Select K-Fold
            st.subheader(f"{get_num_section()}.Select K-Fold Cross validation")
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
            # Select metric
            st.subheader(f"{get_num_section()}.Select metrics")
            st.write("Select at least one metric for evaluating model")
            setting.update({"F1":st.checkbox(f"F1",key="F1")})
            setting.update({"LogLoss":st.checkbox(f"LogLoss",key="LogLoss")})
    # Select PCA
    #with st.container():
    #    st.subheader(f"{get_num_section()}.Select PCA")
    #    st.write(f"Do you want to use PCA for reducing demensions")
    #    pca = st.checkbox(f"PCA",key="PCA")
    #    if pca:
    #        if n_selected_features<2:
    #            st.error("Please select at lease a feature")
    #        elif (setting["F1"] or setting["LogLoss"])==False:
    #            st.error("Please select a metric for evaluation")
    #        else:
    #            feature_selected = [ i for i in get_feature_inputs.keys() if get_feature_inputs[i]]
    #            setting.update({"feature_list":feature_selected,"target":target})
    #            
    #            fig, best = ml.search_PCA(dataset, setting, n_selected_features)
    #            st.pyplot(fig)
    #            reduce = st.slider(f"Select components bewteen 1 and {n_selected_features}",key="pca_n", step=1, min_value=1, max_value = int(n_selected_features), value=1)
    #            setting.update({"pca_n":reduce})
    setting.update({"pca":False})
    
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
                best_model={"f1":None,"LogLoss":None}
                best_value={"f1":0,"LogLoss":float("INF")}

                for modeli in stats["f1"].keys():
                    if stats["f1"][modeli] > best_value["f1"]:
                        best_value["f1"] = stats["f1"][modeli]
                        best_model["f1"] = modeli

                for modeli in stats["LogLoss"].keys():
                    if stats["LogLoss"][modeli] < best_value["LogLoss"]:
                        best_value["LogLoss"] = stats["LogLoss"][modeli]
                        best_model["LogLoss"] = modeli
                c1,c2 = st.columns(2)
                with c1:
                    st.metric(f"F1 Score - {best_model['f1']}", "{v:.2f}".format(v=best_value["f1"]))
                with c2:
                    st.metric(f"Log Loss - {best_model['LogLoss']}", "{v:.2f}".format(v=best_value["LogLoss"]))
                st.pyplot(model.plot_history())