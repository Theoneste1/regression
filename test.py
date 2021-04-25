# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 10:49:39 2020

@author: user
"""

import os
import base64
import streamlit as st
import pickle

st.title('Downloader')


# Core Pkgs
import streamlit as st 
import streamlit.components.v1 as stc 
import streamlit as st
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os


html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Machine Learning with Regression</h1>
		<h4 style="color:white;text-align:center;">Enjoy using Any algorithm</h4>
		</div>
		"""
stc.html(html_temp)

st.write("This is a platform that will the people who are not familiar with writing the codes to do machine learning by using their choosen model")
st.write("This will help the people to run the different models, predict without needging to have the skills about how to write the codes")

st.subheader("START BY SELECTING THE DATASET TO USE")

st.write("You will first need to use the dataset that you would like to use")


def file_selector(folder_path='./data'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("SELECT THE DATASET TO USE FROM BELLOW BOX",filenames)
    return os.path.join(folder_path,selected_filename)

filename = file_selector()
st.info("YOU SELECTED TO USE: {}".format(filename))

# Read Data
df = pd.read_csv(filename)
# Show Dataset



st.subheader("EDA")
submenu = st.sidebar.selectbox("Submenu",["EDA","Plots"])

if submenu == "EDA":
	st.subheader("EXPLORATORY DATASET")
	st.dataframe(df.head())

	c1,c2 = st.beta_columns(2)

	with st.beta_expander("DESCRIPTIVE SUMMARY"):
		st.dataframe(df.describe())
        
status = st.radio("DO YOU WANT TO VISUALIE NULL VALUES?",("yes","no"))

if status == 'yes':
	st.write(df.isnull().sum())
else:
	st.warning("Inactive, Activate")
    
#if st.checkbox("Clean your Data by removing the null values"):
#    df.dropna(axis=1, how='all')
#    st.dataframe(df.head())

st.subheader("CUSTOMIZABLE PLOT")

st.write("If you need to visualize the dataset on plot, please remember to choose the fields that you want to use.")
st.write("Remember that the fields that you will choose are the one which will appear on plot you are going to do")

st.subheader("YOU WILL NEED TO CHOOSE THE FIELDS OR COLUMNS YOU WANT TO USE IN PLOTING")
all_columns_names = df.columns.tolist()
selected_columns_names = st.multiselect("Select the fields To Plot",all_columns_names)

st.subheader("YOU WILL NEED TO CHOOSE THE TYPES OF PLOT YOU WANT TO CHECK WITH")

st.write("We  have different kind of plot that you can use, so here you will need to choose the one that you will go with")
type_of_plot = st.selectbox("SELECT PLOT HERE PLEASE",["area","bar","line","hist","box","kde"])


if st.button("Generate Plot"):
    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

    # Plot By Streamlit
if type_of_plot == 'area':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    cust_data = df[selected_columns_names]
elif type_of_plot == 'bar':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    cust_data = df[selected_columns_names]
    st.bar_chart(cust_data)
elif type_of_plot == 'line':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    cust_data = df[selected_columns_names]
    st.line_chart(cust_data)

# Custom Plot 
elif type_of_plot:
  cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
  st.write(cust_plot)
  st.pyplot()
  
st.subheader("DATA CLEANING AREA")
st.write('Here you will be able to clean your dataset')
  
if st.checkbox("Visualize null value"):
    st.dataframe(df.isnull().sum())
if st.checkbox("Visualize categorical features"):
    categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
    dt=df[categorical_feature_columns]
    st.dataframe(dt)
if st.checkbox("Encoding features"):
    label= LabelEncoder()
    for col in df.columns:
        df[col]=label.fit_transform(df[col])
    st.dataframe(df)
  

    
st.subheader("TRAINING THE MODEL BY USING YOUR CHOISE ALGORITHM")
st.write("Here you will need to choose the fields that you will use to train your model.")
st.write("Please, remember that in the fields that you will select, there should be a target one, so it is a must to have it.")
  
if st.checkbox("Select the features"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select",all_columns)
    new_df = df[selected_columns]
    st.dataframe(new_df)
    df=new_df

 
        
#encoding the things // data transformation
label= LabelEncoder()
for col in df.columns:
    df[col]=label.fit_transform(df[col])
Y = df.target
X = df.drop(columns=['target'])
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)


st.subheader("SCALE THE DATASET")

if st.checkbox("Scale data"):
  sl=StandardScaler()
  X_trained= sl.fit_transform(X_train)
  X_tested= sl.fit_transform(X_test)
  st.dataframe(X_trained)
  
#models
classifier_name = st.selectbox(
      'Machine Learning Algorithm',
      ('Linear Regression', 'SVR','Lasso Regression','Decision Tree','GradientBoostingRegressor','AdaBoostRegressor')
  )

#if model is Linear Regression
if classifier_name == 'Linear Regression':
    st.subheader('Hyperparmeter tuning')
    n_jobs= st.number_input("number of jops",1,10,step=1,key='jobs')
    normalize= st.radio("normalize",("False","True"),key='normilize')
    if st.button("classify",key='classify'):
        st.subheader("Linear Regression result")
        linear= LinearRegression(normalize=normalize,n_jobs=n_jobs)
        linear.fit(X_train,y_train)
        preds= linear.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Root Mean squared error: %.2f" % mean_squared_error(y_test,preds))
        st.write("Mean Absolute error: %.2f" % mean_absolute_error(y_test,preds))
        filename = 'finalized_model.sav'
        pickle.dump(linear, open(filename, 'wb'))
        
#if model is svr
if classifier_name == 'SVR':
    st.subheader('Cross Validation')
    if st.checkbox("KFold"):
       n_splits= st.slider("number of split",min_value=1, max_value =10,step=1,key='split')
       kfold= KFold(n_splits=n_splits)
      
    if st.button("classify",key='classify'):
         score =  cross_val_score(SVR(),X,Y,cv=kfold)
         st.write("SVR score:",score.mean())
       
              

#if model is Lasso
if classifier_name == 'Lasso Regression':
    if st.button("classify",key='classify'):
        st.subheader("Lasso Regression result")
        lassomodel= Lasso()
        lassomodel.fit(X_train,y_train)
        preds= lassomodel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Root Mean squared error: %.2f" % mean_squared_error(y_test,preds))
        st.write("Mean Absolute error: %.2f" % mean_absolute_error(y_test,preds))

#if model is decision
if classifier_name == 'Decision Tree':
    if st.button("classify",key='classify'):
        st.subheader("Decision  result")
        decisionModel= DecisionTreeRegressor()
        decisionModel.fit(X_train,y_train)
        preds= decisionModel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Root Mean squared error: %.2f" % mean_squared_error(y_test,preds))
        st.write("Mean Absolute error: %.2f" % mean_absolute_error(y_test,preds))


#if model is GradientBoostingRegressor
if classifier_name == 'GradientBoostingRegressor':
    if st.button("classify",key='classify'):
        st.subheader("GradientBoostingRegressor Result")
        gradientModel= GradientBoostingRegressor()
        gradientModel.fit(X_train,y_train)
        preds= gradientModel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Root Mean squared error: %.2f" % mean_squared_error(y_test,preds))
        st.write("Mean Absolute error: %.2f" % mean_absolute_error(y_test,preds))


#if model is AdaBoostRegressor
if classifier_name == 'AdaBoostRegressor':
    st.subheader('Hyperparmeter tuning')
    n_estimators= st.slider("number of estimators",min_value=1, max_value =10,step=1,key='jobs')
    loss= st.radio("Loss",("linear", "square", "exponential"),key='loss')
    learning_rate= st.number_input("Learning rate",1,10,step=1,key='late')
    if st.button("classify",key='classify'):
        st.subheader("AdaBoostRegressor Result")
        AdaBoostRegressorModel= AdaBoostRegressor(n_estimators=n_estimators,loss=loss,learning_rate=learning_rate)
        AdaBoostRegressorModel.fit(X_train,y_train)
        preds= AdaBoostRegressorModel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Root Mean squared error: %.2f" % mean_squared_error(y_test,preds))
        st.write("Mean Absolute error: %.2f" % mean_absolute_error(y_test,preds))


st.subheader("PREDICT BY USING ADDBOOST MODEL")    
#    prediction part    
if st.checkbox('prediction'):
    dt= set(X.columns)
    user_input=[]
    for i in dt:
        firstname = st.text_input(i,"Type here...")
        user_input.append(firstname)
    my_array= np.array([user_input])
    AdaBoostRegressorModel= AdaBoostRegressor()
    AdaBoostRegressorModel.fit(X_train,y_train)
    y_user_prediction= AdaBoostRegressorModel.predict(my_array)
    st.write(y_user_prediction)
    


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.markdown(get_binary_file_downloader_html('finalized_model.sav', 'Text Download'), unsafe_allow_html=True)