import streamlit as st
import pandas as pd
import joblib
import os

# 🎯 Page config
st.set_page_config(page_title="WellMind Predictor", page_icon="🧠", layout="centered")

# 🌟 Title and description
st.markdown("""
    <h1 style='text-align:center; color:#4b8bbe;'>🧠 WellMind</h1>
    <h3 style='text-align:center; color:gray;'>Mental Health Risk Prediction Tool</h3>
    <p style='text-align:center; font-size:16px;'>Know when to care. Predict your mental well-being with smart tech.</p>
""", unsafe_allow_html=True)

# ✅ Load model
model_path = "mental_health_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 🧾 Collect input with columns
st.markdown("### 📋 Fill out your information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 65, 30)
    gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
    family_history = st.selectbox("🧬 Family history of mental illness?", ["Yes", "No"])

with col2:
    work_interfere = st.selectbox("💼 Mental health affects your work?", ["Often", "Sometimes", "Rarely", "Never"])
    benefits = st.selectbox("🏥 Employer offers mental health benefits?", ["Yes", "No", "Don't know"])

# 🧮 Manual encoding
input_dict = {
    "Age": age,
    "Gender": {"Male": 1, "Female": 0, "Other": 2}[gender],
    "family_history": {"Yes": 1, "No": 0}[family_history],
    "work_interfere": {"Often": 3, "Sometimes": 1, "Rarely": 2, "Never": 0}[work_interfere],
    "benefits": {"Yes": 2, "No": 0, "Don't know": 1}[benefits],
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Match expected model features
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# 🎯 Predict
st.markdown("---")
if st.button("🔍 Predict My Mental Health Risk"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🔴 You may need mental health support.")
        st.markdown("""
        <div style='color: #c0392b; font-size: 16px;'>
        💡 Consider reaching out to a counselor or a trusted support system.<br>
        🧘‍♂️ Take time for self-care, journaling, and breathing exercises.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("🟢 You seem to be doing well mentally!")
        st.markdown("""
        <div style='color: #2ecc71; font-size: 16px;'>
        🎉 Keep taking care of yourself.<br>
        ✅ Healthy routines, social interaction, and regular breaks are powerful!
        </div>
        """, unsafe_allow_html=True)

# 📌 Footer
st.markdown("""
<hr style="margin-top:30px;">
<p style='text-align:center; font-size:13px; color:gray;'>WellMind © 2025 | Mental Health Awareness with AI</p>
""", unsafe_allow_html=True)
