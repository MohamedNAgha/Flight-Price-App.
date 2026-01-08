import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('flight_price_xgb_model.pkl')
    return model

model = load_model()


st.title("✈️ Flight Price Prediction")
st.markdown("Enter flight details to get an estimated ticket price.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    
    airline = st.selectbox(
        "Airline",
        ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 
         'Vistara', 'Air Asia', 'Vistara Premium economy', 'Jet Airways Business', 
         'Multiple carriers Premium economy', 'Trujet']
    )
    

    source = st.selectbox(
        "Source City",
        ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
    )

  
    dep_time = st.time_input("Departure Time")

    
    total_stops = st.selectbox(
        "Total Stops",
        [0, 1, 2, 3, 4],
        format_func=lambda x: "Non-stop" if x == 0 else f"{x} Stop(s)"
    )

with col2:
    
    journey_date = st.date_input("Date of Journey")
    
   
    destination = st.selectbox(
        "Destination City",
        ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']
    )

    
    arrival_time = st.time_input("Arrival Time")
    
   
    duration_h = st.number_input("Duration (Hours)", min_value=0, max_value=50, value=2)
    duration_m = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=30)


if st.button("Predict Price 💸", use_container_width=True):
    
   
    journey_day = journey_date.day
    journey_month = journey_date.month
    dep_hour = dep_time.hour
    dep_min = dep_time.minute
    arrival_hour = arrival_time.hour
    arrival_min = arrival_time.minute
    duration_total_mins = (duration_h * 60) + duration_m

    
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Total_Stops': [total_stops],
        'Journey_Day': [journey_day],
        'Journey_Month': [journey_month],
        'Dep_Hour': [dep_hour],
        'Dep_Min': [dep_min],
        'Arrival_Hour': [arrival_hour],
        'Arrival_Min': [arrival_min],
        'Duration_Total_Mins': [duration_total_mins]
    })

    try:
        
        log_price = model.predict(input_data)[0]
        
        original_price = np.expm1(log_price)
        
        st.success(f"Estimated Price: **₹ {original_price:,.2f}**")
        
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")