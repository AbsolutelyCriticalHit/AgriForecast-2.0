import streamlit as st 
import joblib
import numpy as np

# ✅ Load your trained model
model = joblib.load('model_rf.pkl')

# ✅ App title
st.title("🌾 Crop Yield Prediction App")

# ✅ Input fields
rain = st.number_input("Average Rainfall (mm/year)", min_value=0.0)
temp = st.number_input("Average Temperature (°C)", min_value=0.0)
pest = st.number_input("Pesticides Used (tonnes)", min_value=0.0)

# ✅ Use the EXACT names from model_columns
item = st.selectbox("Crop Type", [
    'cassava',
    'maize',
    'potatoes',
    'rice, paddy',
    'soybeans',
    'sweet potatoes'
])
area = st.selectbox("Country", ['indonesia', 'others'])

# ✅ Model columns used during training
model_columns = [
    'Area',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
    'Item_cassava',
    'Item_maize',
    'Item_potatoes',
    'Item_rice, paddy',
    'Item_soybeans',
    'Item_sweet potatoes'
]

# ✅ Prepare input
input_data = {col: 0 for col in model_columns}
input_data['average_rain_fall_mm_per_year'] = np.log(rain + 1)
input_data['avg_temp'] = np.log(temp + 1)
input_data['pesticides_tonnes'] = np.log(pest + 1)
input_data[f'Item_{item}'] = 1  # e.g. Item_rice, paddy
input_data['Area'] = 1 if area.lower() == 'indonesia' else 0

# ✅ Create final input array
X = np.array([input_data[col] for col in model_columns]).reshape(1, -1)

# ✅ Prediction
if st.button("Predict", key="predict_button"):
    prediction = model.predict(X)[0]
    st.success(f"🌾 Predicted Yield: {prediction:.2f} hg/ha")
