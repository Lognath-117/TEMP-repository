import streamlit as st
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie

# ------------------- CONFIGURATION -------------------
st.set_page_config(page_title="WellMind 2.0", page_icon="🧠", layout="centered")
st.markdown("<style>body {background-color: #f0f4f8;}</style>", unsafe_allow_html=True)

# ------------------- LOAD ANIMATION -------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_mental = load_lottie_url("https://lottie.host/4ef6b5c7-99ce-471c-9670-e1f0dcf48ec5/vzLrTxFn0K.json")

# ------------------- LOAD MODEL -------------------
model_path = "mental_health_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ------------------- HEADER -------------------
with st.container():
    st_lottie(lottie_mental, speed=1, height=200, key="mental")
    st.markdown("""
        <h1 style='text-align:center; color:#4b8bbe;'>🧠 WellMind 2.0</h1>
        <p style='text-align:center; color:gray; font-size:18px;'>AI-Powered Mental Health Risk Prediction Tool</p>
    """, unsafe_allow_html=True)

# ------------------- INPUT FORM -------------------
st.markdown("### 📋 Let's understand you better:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 65, 30)
    gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
    family_history = st.selectbox("🧬 Family history of mental illness?", ["Yes", "No"])

with col2:
    work_interfere = st.selectbox("💼 Mental health affects your work?", ["Often", "Sometimes", "Rarely", "Never"])
    benefits = st.selectbox("🏥 Employer offers mental health benefits?", ["Yes", "No", "Don't know"])

# ------------------- ENCODING INPUT -------------------
input_dict = {
    "Age": age,
    "Gender": {"Male": 1, "Female": 0, "Other": 2}[gender],
    "family_history": {"Yes": 1, "No": 0}[family_history],
    "work_interfere": {"Often": 3, "Sometimes": 1, "Rarely": 2, "Never": 0}[work_interfere],
    "benefits": {"Yes": 2, "No": 0, "Don't know": 1}[benefits],
}
input_df = pd.DataFrame([input_dict])

# Match expected model features
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# ------------------- PREDICTION -------------------
st.markdown("### 🔍 Prediction Result")

if st.button("🧠 Analyze My Mental Health Risk"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]  # Confidence score

    # Show result
    if prediction == 1:
        st.error("🔴 You may need mental health support.")
        st.markdown(f"**Confidence:** {proba*100:.2f}%")
        st.markdown("""
        <div style='color: #c0392b; font-size: 16px;'>
        💬 You're not alone. Many people experience mental health challenges.<br>
        🔗 <a href="https://www.mind.org.uk/information-support/" target="_blank">Explore help options →</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("🟢 You appear to be in a good mental health state.")
        st.markdown(f"**Confidence:** {proba*100:.2f}%")
        st.markdown("""
        <div style='color: #2ecc71; font-size: 16px;'>
        🌱 Keep nurturing your mental wellness.<br>
        ✨ Practice mindfulness, maintain work-life balance.
        </div>
        """, unsafe_allow_html=True)

    # Show summary of inputs
    st.markdown("---")
    st.markdown("#### 📝 Summary of Your Input")
    st.dataframe(pd.DataFrame(input_dict, index=["Your Input"]).T)

# ------------------- FOOTER -------------------
st.markdown("""
<hr>
<p style='text-align:center; font-size:13px; color:#888;'>Made with 💙 by WellMind AI | Mental Health Matters</p>
""", unsafe_allow_html=True)
