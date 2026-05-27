import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
st.title("🌾 Crop Yield Prediction System")
st.write("Input the environmental and agricultural factors to estimate the crop yield.")

# --- Data Loading & Model Training ---
@st.cache_resource
def load_and_train():
    # Load your dataset
    df = pd.read_csv('crop_yield.csv')
    df = df.drop(['Crop_Year'], axis=1, errors='ignore')
    
    # We need to keep the encoders to transform user input later
    le_crop = LabelEncoder()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    
    # Store the original names for the dropdown menus
    crop_list = sorted(df['Crop'].unique())
    state_list = sorted(df['State'].unique())
    season_list = sorted(df['Season'].unique())
    
    # Fit and Transform
    df['Crop'] = le_crop.fit_transform(df['Crop'])
    df['State'] = le_state.fit_transform(df['State'])
    df['Season'] = le_season.fit_transform(df['Season'])
    
    # Define Features and Target
    X = df.drop(['Yield'], axis=1)
    y = df['Yield']
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    
    return model, le_crop, le_state, le_season, crop_list, state_list, season_list, X.columns

# Initialize everything
model, le_crop, le_state, le_season, crops, states, seasons, feature_cols = load_and_train()

# --- User Interface ---
st.sidebar.header("Input Agricultural Data")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        selected_crop = st.selectbox("Select Crop", crops)
        selected_state = st.selectbox("Select State", states)
        selected_season = st.selectbox("Select Season", seasons)
        area = st.number_input("Area (Hectares)", min_value=0.0, value=1000.0)
        
    with col2:
        production = st.number_input("Production (Tonnes)", min_value=0.0, value=5000.0)
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)
        fertilizer = st.number_input("Fertilizer Usage (kg)", min_value=0.0, value=200000.0)
        pesticide = st.number_input("Pesticide Usage (kg)", min_value=0.0, value=500.0)

# --- Prediction Logic ---
if st.button("Predict Crop Yield", type="primary"):
    # Encode user selections using the pre-trained encoders
    input_data = pd.DataFrame({
        'Crop': [le_crop.transform([selected_crop])[0]],
        'Season': [le_season.transform([selected_season])[0]],
        'State': [le_state.transform([selected_state])[0]],
        'Area': [area],
        'Production': [production],
        'Annual_Rainfall': [rainfall],
        'Fertilizer': [fertilizer],
        'Pesticide': [pesticide]
    })
    
    # Ensure column order matches training data
    input_data = input_data[feature_cols]
    
    prediction = model.predict(input_data)
    
    st.markdown("---")
    st.success(f"### Estimated Yield: {prediction[0]:.2f} quintal/hectare")
    
    # Visual feedback
    st.info("Yield is calculated based on production relative to the area and environmental factors.")
    
