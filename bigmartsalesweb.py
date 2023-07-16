# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 17:16:20 2023

@author: User
"""

import numpy as np
import pickle
import streamlit as st



loaded_model=pickle.load(open(r"C:\Users\User\Desktop\big mart sales\trained_model.sav","rb"))
def bigmartsales(input_data):
    input_data = [float(x) for x in input_data]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    
    st.title('Big Mart Sales')
    
    										
    Item_Identifier=st.text_input('Item Identification')
    Item_Weight=st.text_input('Item Weight')
    Item_Fat_Content=st.text_input('Item fat content')
    Item_Visibility=st.text_input('Item visibility')
    Item_Type=st.text_input('Item type')
    Item_MRP=st.text_input('Item MRP')
    Outlet_Identifier=st.text_input('outlet item')
    Outlet_Establishment_Year=st.text_input('Outlet Establishment Year')
    Outlet_Size=st.text_input('Outlet SIze')
    Outlet_Location_Type=st.text_input('Outlet Location Type')
    Outlet_Type	=st.text_input('Outlet Type')

    sales=''
    
    if st.button('Calculate'):
       sales= bigmartsales([ Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year, Outlet_Size,Outlet_Location_Type, Outlet_Type])
      
    st.success(sales)
    
if __name__=='__main__':
    main()    
    
    
    
    
    
    
