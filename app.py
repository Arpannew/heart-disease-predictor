import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("heart_model.joblib")

st.title("🫀 Heart Disease Predictor")
st.markdown("Enter patient details in the sidebar to predict heart disease.")

# Sidebar input fields
age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ['M', 'F'])
cp = st.sidebar.selectbox("Chest Pain Type", ['ASY', 'NAP', 'ATA', 'TA'])
restbp = st.sidebar.number_input("Resting BP", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fastingbs = st.sidebar.selectbox("Fasting BS > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
maxhr = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# Input as dictionary
input_data = {
    'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': restbp,
    'Cholesterol': chol, 'FastingBS': fastingbs, 'RestingECG': restecg,
    'MaxHR': maxhr, 'ExerciseAngina': exang, 'Oldpeak': oldpeak, 'ST_Slope': slope
}

if st.button("🔍 Predict"):
    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    st.subheader("Result:")
    st.write(f"Prediction: {'🚨 Heart Disease' if pred == 1 else '✅ No Heart Disease'}")
    st.write(f"Probability: {round(prob*100, 2)}%")
