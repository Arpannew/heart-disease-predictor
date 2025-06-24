# import streamlit as st
# import pandas as pd
# import joblib

# # Load the model
# model = joblib.load("heart_model_joblib.ipynb")

# st.title("🫀 Heart Disease Predictor")
# st.markdown("Enter patient details in the sidebar to predict heart disease.")

# # Sidebar input fields
# age = st.sidebar.number_input("Age", 20, 100, 50)
# sex = st.sidebar.selectbox("Sex", ['M', 'F'])
# cp = st.sidebar.selectbox("Chest Pain Type", ['ASY', 'NAP', 'ATA', 'TA'])
# restbp = st.sidebar.number_input("Resting BP", 80, 200, 120)
# chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
# fastingbs = st.sidebar.selectbox("Fasting BS > 120 mg/dl", [0, 1])
# restecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
# maxhr = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
# exang = st.sidebar.selectbox("Exercise Induced Angina", ['Y', 'N'])
# oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
# slope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# # Input as dictionary
# input_data = {
#     'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': restbp,
#     'Cholesterol': chol, 'FastingBS': fastingbs, 'RestingECG': restecg,
#     'MaxHR': maxhr, 'ExerciseAngina': exang, 'Oldpeak': oldpeak, 'ST_Slope': slope
# }

# if st.button("🔍 Predict"):
#     df = pd.DataFrame([input_data])
#     pred = model.predict(df)[0]
#     prob = model.predict_proba(df)[0][1]
    
#     st.subheader("Result:")
#     st.write(f"Prediction: {'🚨 Heart Disease' if pred == 1 else '✅ No Heart Disease'}")
#     st.write(f"Probability: {round(prob*100, 2)}%")


import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor")
st.title("🩺 Heart Disease Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=54)
    sex = st.sidebar.selectbox("Sex", ['M', 'F'])
    cp = st.sidebar.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
    restingbp = st.sidebar.number_input("Resting BP", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("FastingBS > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
    maxhr = st.sidebar.number_input("Max Heart Rate", 60, 210, 150)
    exang = st.sidebar.selectbox("Exercise Angina", ['Y', 'N'])
    oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

    data = {
        'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': restingbp,
        'Cholesterol': chol, 'FastingBS': fbs, 'RestingECG': restecg,
        'MaxHR': maxhr, 'ExerciseAngina': exang, 'Oldpeak': oldpeak,
        'ST_Slope': slope
    }
    return pd.DataFrame([data])

input_df = user_input()

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🧾 Prediction Result")
    st.write("✅ **No Heart Disease Detected**" if prediction == 0 else "⚠️ **Heart Disease Detected!**")
    st.write(f"Probability of heart disease: **{prob:.2f}**")
