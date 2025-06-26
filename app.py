import streamlit as st 
import joblib
import numpy as np

model = joblib.load('model_rf.pkl')

st.title("Crop Yield Prediction App (Indonesia Only)")

rain = st.number_input("Average Rainfall (mm/year)", min_value=0.0)
temp = st.number_input("Average Temperature (°C)", min_value=0.0)
pest = st.number_input("Pesticides Used (tonnes)", min_value=0.0)

item = st.selectbox("Crop Type", [
    'cassava',
    'maize',
    'potatoes',
    'rice, paddy',
    'soybeans',
    'sweet potatoes'
])

item_map = {
  'cassava': 'Item_cassava',
  'maize': 'Item_maize',
  'potatoes': 'Item_potatoes',
  'rice, paddy': 'Item_rice_paddy',
  'soybeans': 'Item_soybeans',
  'sweet potatoes': 'Item_sweet_potatoes'
}
selected = item_map[item]
input_data[selected] = 1


input_data = {col: 0 for col in model_columns}
input_data['average_rain_fall_mm_per_year'] = np.log(rain + 1)
input_data['avg_temp'] = np.log(temp + 1)
input_data['pesticides_tonnes'] = np.log(pest + 1)
input_data[f'Item_{item}'] = 1
input_data['Area'] = 1  # Hardcoded to Indonesia

X = np.array([input_data[col] for col in model_columns]).reshape(1, -1)

if st.button("Predict", key="predict_button"):
    # Show expected features from the model
    st.write("🧠 Model expects these features:", model.feature_names_in_)
    
    # Show what is being passed in
    st.write("📊 Model input array:", X)

    # Make prediction
    prediction = model.predict(X)[0]
    st.success(f"🌾 Predicted Yield: {prediction:.2f} hg/ha")

