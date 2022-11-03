import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml

#st.title("My First Dang's Web")

#1. Markdown
st.markdown("""
# Hệ thống AI dự báo
## 1. Dự báo lương
""")
data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="data")
setting={}
if data is not None:
    byte_data=data.getvalue()
    with open(f"./data/{data.name}","wb+") as f:
        f.write(byte_data)
    
    # Select input feature
    dataset = ml.Dataset(f"./data/{data.name}")
    st.dataframe(dataset.origin_data)
    get_feature_inputs={}
    target = dataset.origin_data.columns[-1]
    st.write("Select features as input features")
    for i in list(dataset.origin_data.columns)[:-1]:
        get_feature_inputs.update({i:st.checkbox(f"{i}",key=f"feature_select_{i}")})
    
    # Select output feature
    st.write(f"Output features: {target}")

    # Select K-Fold
    kfold = st.checkbox(f"K-Fold Cross validation",key="kfold")
    setting.update({"kfold":kfold})
    if kfold:
        K = st.number_input("Enter K",key="rate_train", step=1, min_value=2, value=2)
        setting.update({"K":K})

    # Select the rate of train dataset
    if kfold == False:
        rate = st.number_input("Enter the rate of train dataset",key="rate_train", value=0.1, min_value=0.1, max_value=1.0, step=0.01)
        setting.update({"rate":rate})

    # Select metric
    st.write("Select at least one metric for evaluating model")
    setting.update({"F1":st.checkbox(f"F1",key="F1")})
    setting.update({"Acc":st.checkbox(f"Acc",key="Acc")})
    
    if st.button("Run",key="run"):
        if (setting["F1"] or setting["Acc"])==False:
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