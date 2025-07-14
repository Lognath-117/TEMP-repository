import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Simple Mental Health Predictor", page_icon="🧠")

st.title("🧠 Simple Mental Health Risk Checker")

# ✅ Load model safely
model_path = "mental_health_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please make sure 'mental_health_model.pkl' is in the same folder.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 🎛️ Minimal Inputs (only most impactful)
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
work_interfere = st.selectbox("Does mental health affect your work?", ["Often", "Sometimes", "Rarely", "Never"])
benefits = st.selectbox("Does your employer offer mental health benefits?", ["Yes", "No", "Don't know"])

# 🧮 Simple encoding (same as during training)
input_dict = {
    "Age": age,
    "Gender": {"Male": 1, "Female": 0, "Other": 2}[gender],
    "family_history": {"Yes": 1, "No": 0}[family_history],
    "work_interfere": {"Often": 3, "Sometimes": 1, "Rarely": 2, "Never": 0}[work_interfere],
    "benefits": {"Yes": 2, "No": 0, "Don't know": 1}[benefits],
}

# Match model columns
input_df = pd.DataFrame([input_dict])

# Handle missing model features
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# 🧠 Predict
if st.button("Predict Mental Health Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🔴 You may need mental health support.")
    else:
        st.success("🟢 You're unlikely to need mental health support.")
