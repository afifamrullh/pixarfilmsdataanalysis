import streamlit as st
import numpy as np
import joblib

#Load Model
model = joblib.load('diabetes_model.pkl')

#Judul Aplikasi
st.title("Prediksi Diabetes")

# Form input
with st.form("form_diabetes"):
  pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
  glucose = st.number_input('Glucose', min_value=0, max_value=200)
  blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150)
  skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
  insulin = st.number_input('Insulin', min_value=0, max_value=1000)
  bmi = st.number_input('BMI', min_value=0.0, max_value=70.0)
  dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5)
  age = st.number_input('Age', min_value=1, max_value=120)
  submit = st.form_submit_button("Proses")
# Ketika tombol ditekan
if submit:
  # Format input ke bentuk array
  features = np.array([[pregnancies, glucose,
  blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
  # Prediksi
  prediction = model.predict(features)[0]
  # Tampilkan hasil
  if prediction == 1:
    st.error("Hasil: Positif Diabetes")
  else:
    st.success("Hasil: Tidak Diabetes")
