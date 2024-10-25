import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
from joblib import load
import pandas as pd
from pathlib import Path




# Sidebar navigation
with st.sidebar:
    selected = option_menu('Tutorial Desain Streamlit UTS ML 24/25', 
                           ['Klasifikasi', 'Regresi', 'Catatan'],
                           default_index=0)

# Classification section
if selected == 'Klasifikasi':
    model_path = Path('C:/Users/ACER/Downloads/UTS MESIN/BestModel_GBC_SVM_Transformers.pkl')
    st.title('Category Classification')

    # Input fields for features
    st.header("Input Features")
    
    house_size = st.slider("House Size (in sqm)", 50, 1000, step=10)
    number_of_rooms = st.slider("Number of Rooms", 1, 15, step=1)
    has_yard = st.selectbox("Has Yard?", ["Yes", "No"])
    has_pool = st.selectbox("Has Pool?", ["Yes", "No"])
    is_new_built = st.selectbox("Is New Built?", ["Yes", "No"])
    has_storm_protector = st.selectbox("Has Storm Protector?", ["Yes", "No"])
    has_storage_room = st.selectbox("Has Storage Room?", ["Yes", "No"])

    # Additional inputs
    floors = st.slider("Number of Floors", 1, 10, step=1)
    citycode = st.number_input("City Code", min_value=0, max_value=999)
    citypartrange = st.slider("City Part Range", 1, 5, step=1)
    numprevowners = st.slider("Number of Previous Owners", 0, 10, step=1)
    made = st.slider("Year Built", 1900, 2023, step=1)
    basement = st.selectbox("Has Basement?", ["Yes", "No"])
    attic = st.selectbox("Has Attic?", ["Yes", "No"])
    garage = st.selectbox("Has Garage?", ["Yes", "No"])
    has_guest_room = st.selectbox("Has Guest Room?", ["Yes", "No"])

    # Mock classification logic
    def mock_classify(input_data):
        # Simple rules for classification
        if input_data['squaremeters'] > 700 and input_data['numberofrooms'] > 4:
            return "Luxury"
        elif input_data['hasyard'] and input_data['haspool']:
            return "Luxury"
        elif input_data['numberofrooms'] < 3:
            return "Basic"
        else:
            return "Middle"

    # Prepare input data for prediction
    input_data = {
        'squaremeters': house_size,
        'numberofrooms': number_of_rooms,
        'hasyard': 1 if has_yard == "Yes" else 0,
        'haspool': 1 if has_pool == "Yes" else 0,
        'floors': floors,
        'citycode': citycode,
        'citypartrange': citypartrange,
        'numprevowners': numprevowners,
        'made': made,
        'isnewbuilt': 1 if is_new_built == "Yes" else 0,
        'hasstormprotector': 1 if has_storm_protector == "Yes" else 0,
        'basement': 1 if basement == "Yes" else 0,
        'attic': 1 if attic == "Yes" else 0,
        'garage': 1 if garage == "Yes" else 0,
        'hasstorageroom': 1 if has_storage_room == "Yes" else 0,
        'hasguestroom': 1 if has_guest_room == "Yes" else 0,
    }

    if st.button("Classify"):
        label = mock_classify(input_data)
        st.write(f"Predicted Category: {label}")  

if selected == 'Regresi':
    model_path = "C:/Users/ACER/Downloads/UTS MESIN/BestModel_REG_LassoRegression_Transformers.pkl"
    with open(model_path, 'rb') as file:
        full_model = pickle.load(file)
    st.title('Price Prediction using Ridge Regression')

    st.header("Input Features")
    
    house_size = st.slider("House Size (in sqm)", 50, 1000, step=10)
    number_of_rooms = st.slider("Number of Rooms", 1, 15, step=1)
    building_age = st.slider("Building Age (years)", 0, 100, step=1)
    has_yard = st.selectbox("Has Yard?", ["Yes", "No"])
    has_pool = st.selectbox("Has Pool?", ["Yes", "No"])
    is_new_built = st.selectbox("Is New Built?", ["Yes", "No"])
    has_storm_protector = st.selectbox("Has Storm Protector?", ["Yes", "No"])
    has_storage_room = st.selectbox("Has Storage Room?", ["Yes", "No"])

    floors = st.slider("Number of Floors", 1, 10, step=1)
    citycode = st.number_input("City Code", min_value=0, max_value=999)
    citypartrange = st.slider("City Part Range", 1, 5, step=1)
    numprevowners = st.slider("Number of Previous Owners", 0, 10, step=1)
    made = st.slider("Year Built", 1900, 2023, step=1)
    basement = st.selectbox("Has Basement?", ["Yes", "No"])
    attic = st.selectbox("Has Attic?", ["Yes", "No"])
    garage = st.selectbox("Has Garage?", ["Yes", "No"])
    has_guest_room = st.selectbox("Has Guest Room?", ["Yes", "No"])
    
    has_yard = 1 if has_yard == "Yes" else 0
    has_pool = 1 if has_pool == "Yes" else 0
    is_new_built = 1 if is_new_built == "Yes" else 0
    has_storm_protector = 1 if has_storm_protector == "Yes" else 0
    has_storage_room = 1 if has_storage_room == "Yes" else 0
    basement = 1 if basement == "Yes" else 0
    attic = 1 if attic == "Yes" else 0
    garage = 1 if garage == "Yes" else 0
    has_guest_room = 1 if has_guest_room == "Yes" else 0

    input_data = np.array([[house_size, number_of_rooms, has_yard, has_pool, floors, citycode,
                            citypartrange, numprevowners, made, is_new_built, has_storm_protector, 
                            basement, attic, garage, has_storage_room, has_guest_room, building_age]])


    st.write("Input data shape:", input_data.shape)

    if st.button("Predict Price"):
        try:
            predicted_price = full_model.predict(input_data)
            st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")
        except ValueError as e:
            st.error(f"Prediction error: {e}")
