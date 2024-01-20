# importing the necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle


# reading the csv file
df = pd.read_csv("cleaned_copper_dataset.csv")


# function to predict status ("won" or "lost")
def predict_status(quantity_tons, customer, country, item_type, application, thickness,
                   width, product_ref, item_date, delivery_date, selling_price):
    
    global df

    item_date = item_date.strftime("%Y/%m/%d")
    delivery_date = delivery_date.strftime("%Y/%m/%d")
    
    item_date_year = item_date.split("/")[0]
    item_date_month = item_date.split("/")[1]
    item_date_day = item_date.split("/")[2]

    delivery_date_year = delivery_date.split("/")[0]
    delivery_date_month = delivery_date.split("/")[1]
    delivery_date_day = delivery_date.split("/")[2]

    item_type_ohe = pd.get_dummies(df["item type"])
    item_type = item_type_ohe.loc[item_type_ohe.loc[:,f"{item_type}"] == True].values[0].tolist()

    with open("classification_model.pkl", "rb") as file:
        model = pickle.load(file)

    prediction = model.predict([[np.log(quantity_tons), customer, country, np.log(application), np.log(thickness), np.log(width), 
                                 product_ref, item_date_year, item_date_month, item_date_day, delivery_date_year, delivery_date_month, 
                                 delivery_date_day, np.log(selling_price), item_type[0], item_type[1], item_type[2], item_type[3], 
                                 item_type[4], item_type[5], item_type[6] ]])
    
    if prediction[0] == 1:
        return "Won"
    else:
        return "Lost"

    
# function to predict selling price
def predict_selling_price(quantity_tons, customer, country, status, item_type, application,
                          thickness, width, product_ref, item_date, delivery_date):
    
    global df

    item_date = item_date.strftime("%Y/%m/%d")
    delivery_date = delivery_date.strftime("%Y/%m/%d")
    
    item_date_year = item_date.split("/")[0]
    item_date_month = item_date.split("/")[1]
    item_date_day = item_date.split("/")[2]

    delivery_date_year = delivery_date.split("/")[0]
    delivery_date_month = delivery_date.split("/")[1]
    delivery_date_day = delivery_date.split("/")[2]

    status_ohe = pd.get_dummies(df["status"])
    item_type_ohe = pd.get_dummies(df["item type"])

    status = status_ohe.loc[status_ohe.loc[:,f"{status}"] == True].values[0].tolist()
    item_type = item_type_ohe.loc[item_type_ohe.loc[:,f"{item_type}"] == True].values[0].tolist()

    with open("regression_model.pkl", "rb") as file:
        model = pickle.load(file)

    prediction = model.predict([[np.log(quantity_tons), customer, country, np.log(application), np.log(thickness), np.log(width), product_ref, 
                                 item_date_year, item_date_month, item_date_day, delivery_date_year, delivery_date_month, delivery_date_day, 
                                 status[0], status[1], status[2], status[3], status[4], status[5], status[6], status[7], status[8], 
                                 item_type[0], item_type[1], item_type[2], item_type[3], item_type[4], item_type[5], item_type[6] ]])
    
    return np.exp(prediction)

    

# streamlit setup
st.set_page_config("Industrial Copper Modelling", layout = "wide")


selected = option_menu(None,
                       options = ["Menu", "Prediction"],
                       icons = ["house"],
                       orientation = "horizontal",
                       styles = {"nav-link": {"font-size": "18px", "text-align": "center", "margin": "1px"},
                                 "icon": {"color": "yellow", "font-size": "20px"},
                                 "nav-link-selected": {"background-color": "#9457eb"}} )


if selected == "Menu":
    
    st.title(":red[Industrial Copper Modelling]")
    st.markdown("")
    st.markdown('''This project aims to develop two machine learning models for the copper industry to address the challenges of 
                   predicting selling price and lead classification. There are factors like quantity tons, thickness, 
                   application, width etc that will be  very useful for prediction.''')
    
    st.markdown('''* ML Classification model which predicts Status: WON or LOST.''')
    st.markdown('''* ML Regression model which predicts continuous variable ‘Selling_Price’.''')


if selected == "Prediction":

    option = st.selectbox(":violet[**Select Classification or Regression**]", options = ["Classification", "Regression"], key = "option")

    if option == "Classification":

        st.markdown(":violet[**To Predict Status: Won or Lost**]")

        with st.form("cl_model"):

            st.number_input(":blue[**Quantity Tons**]", value = df.loc[0, "quantity tons"], key = "qt1" )

            st.selectbox(":blue[**Customer**]", options = df["customer"].unique(), key = "cust1")

            st.selectbox(":blue[**Country**]", options = df["country"].unique(), key = "cty1")

            st.selectbox(":blue[**Item Type**]", options = df["item type"].unique(), key = "it1")

            st.number_input(":blue[**Application**]", value = df.loc[0, "application"], key = "app1")

            st.number_input(":blue[**Thickness**]",  value = df.loc[0, "thickness"], key = "tkns1")

            st.number_input(":blue[**Width**]", value = df.loc[0, "width"], key = "wth1")

            st.selectbox(":blue[**Product Reference**]", options = df["product_ref"].unique(), key = "pr1")

            st.date_input(":blue[**Item Date**]", key = "id1")

            st.date_input(":blue[**Delivery Date**]", key = "dd1")

            st.number_input(":blue[**Selling Price**]", value = df.loc[0, "selling_price"], key = "sp1" )


            if st.form_submit_button("Predict Status"):

                cl_pred = predict_status(st.session_state["qt1"], st.session_state["cust1"], st.session_state["cty1"], st.session_state["it1"],
                                         st.session_state["app1"], st.session_state["tkns1"], st.session_state["wth1"], st.session_state["pr1"],
                                         st.session_state["id1"], st.session_state["dd1"], st.session_state["sp1"])
                
                st.success(f"The Predicted Status is :green['{cl_pred}']")                    


    if option == "Regression":

        st.markdown(":violet[**To Predict Selling Price**]")

        with st.form("reg_model"):

            st.number_input(":blue[**Quantity Tons**]", value = df.loc[0, "quantity tons"], key = "qt2" )

            st.selectbox(":blue[**Customer**]", options = df["customer"].unique(), key = "cust2")

            st.selectbox(":blue[**Country**]", options = df["country"].unique(), key = "cty2")

            st.selectbox(":blue[**Status**]", options = df["status"].unique(), key = "st2")

            st.selectbox(":blue[**Item Type**]", options = df["item type"].unique(), key = "it2")

            st.number_input(":blue[**Application**]", value = df.loc[0, "application"], key = "app2")

            st.number_input(":blue[**Thickness**]",  value = df.loc[0, "thickness"], key = "tkns2")

            st.number_input(":blue[**Width**]", value = df.loc[0, "width"], key = "wth2")

            st.selectbox(":blue[**Product Reference**]", options = df["product_ref"].unique(), key = "pr2")

            st.date_input(":blue[**Item Date**]", key = "id2")

            st.date_input(":blue[**Delivery Date**]", key = "dd2")


            if st.form_submit_button("Predict Selling Price"):

                reg_pred = predict_selling_price(st.session_state["qt2"], st.session_state["cust2"], st.session_state["cty2"], st.session_state["st2"],
                                                 st.session_state["it2"], st.session_state["app2"], st.session_state["tkns2"], st.session_state["wth2"], 
                                                 st.session_state["pr2"], st.session_state["id2"], st.session_state["dd2"])
                
                st.success(f"The Predicted Selling Price is :green[$ {reg_pred[0]:,.0f}]")
                st.success(f"The Predicted Selling Price is :green[₹ {reg_pred[0]*83:,.0f}]")
                    

# ------------------------------x-------------------------------------x-----------------------------------x-------------------------------------x-----------------------------------------