import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml
import sklearn.datasets as datasets
#st.title("My First Dang's Web")

#1. Markdown
st.markdown("""
# Hệ thống AI dự báo
""")
#data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="data")
setting={}

if True:
#    byte_data=data.getvalue()
#    with open(f"./data/{data.name}","wb+") as f:
#        f.write(byte_data)
    
    # Select input feature
    X,y = datasets.load_wine(return_X_y=True, as_frame=True)
    df = X.copy()
    df["label"]=y
    df = df.sample(frac=1).reset_index(drop=True)

    dataset = ml.Dataset(df,"df")

    st.dataframe(dataset.origin_data)

    get_feature_inputs={}
    target = dataset.origin_data.columns[-1]

    st.write("Select features as input features")
    for i in list(dataset.origin_data.columns)[:-1]:
        get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}")})
    n_selected_features = np.sum(list(get_feature_inputs.values()))
    # Select output feature
    st.write(f"Output features: {target}")

    # Select K-Fold
    kfold = st.checkbox(f"K-Fold Cross validation",key="kfold")
    setting.update({"kfold":kfold})
    if kfold:
        K = st.number_input(f"Enter K between 2 and {n_selected_features}",key="rate_train", step=1, min_value=2, max_value=n_selected_features, value=2)
        setting.update({"K":K})

    # Select the rate of train dataset
    if kfold == False:
        rate = st.number_input("Enter the rate of train dataset",key="rate_train", value=0.1, min_value=0.1, max_value=1.0, step=0.01)
        setting.update({"rate":rate})
    
    # Select PCA
    st.write(f"Do you want to use PCA for reducing demensions")
    pca = st.checkbox(f"PCA",key="PCA")
    if pca:
        if n_selected_features==0:
            st.error("Please select at lease a feature")
        else:
            reduce = st.number_input(f"Enter components bewteen 1 and {n_selected_features}",key="pca_n", step=1, min_value=1, max_value = n_selected_features, value=1)
            setting.update({"pca_n":reduce})
    setting.update({"pca":pca})

    # Select metric
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
                #h = pd.DataFrame(model.history)
                #st.dataframe(h)
                st.pyplot(model.plot_history())
        #for i in get_inputs:
        #    if i not in model_salary.str_col:
        #        get_inputs[i]=float(get_inputs[i])
        ##st.write(f"Input={get_inputs}")
        ##st.write(f"Best model={model.best_model}")
        #st.success(f"Salary = {model_salary.predict(get_inputs)}")