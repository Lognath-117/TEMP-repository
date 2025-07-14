import streamlit as st
import pandas as pd
import joblib
import os

# Setup page
st.set_page_config(page_title="WellMind Pro", page_icon="🧠", layout="centered")

# Load model
model_path = "mental_health_model.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found: mental_health_model.pkl")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main header
st.markdown("""
    <div style="text-align:center">
        <h1 style="color:#4B8BBE;">🧠 WellMind Pro</h1>
        <h4 style="color:gray;">Predict Your Mental Health Risk Instantly</h4>
        <hr style="border:1px solid #eee;">
    </div>
""", unsafe_allow_html=True)

# Input layout
with st.form("mental_health_form"):
    st.subheader("📋 Your Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age", 18, 65, 30)
        gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
        family_history = st.selectbox("🧬 Family history of mental illness?", ["Yes", "No"])

    with col2:
        work_interfere = st.selectbox("💼 Does mental health affect your work?", ["Often", "Sometimes", "Rarely", "Never"])
        benefits = st.selectbox("🏥 Does your employer offer mental health benefits?", ["Yes", "No", "Don't know"])

    submitted = st.form_submit_button("🔍 Predict Risk")

# Encode and Predict
if submitted:
    # Encode input
    input_data = {
        "Age": age,
        "Gender": {"Male": 1, "Female": 0, "Other": 2}[gender],
        "family_history": {"Yes": 1, "No": 0}[family_history],
        "work_interfere": {"Often": 3, "Sometimes": 1, "Rarely": 2, "Never": 0}[work_interfere],
        "benefits": {"Yes": 2, "No": 0, "Don't know": 1}[benefits],
    }
