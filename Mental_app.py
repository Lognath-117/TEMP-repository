
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("mental_health_model.pkl")

# 🎨 Streamlit UI Settings
st.set_page_config(page_title="WellMind - Mental Health Predictor", page_icon="🧠", layout="centered")

# 🧠 Title & Header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4b8bbe;'>🧠 WellMind</h1>
        <h3 style='color: #333;'>Mental Health Risk Prediction System</h3>
        <p style='color: gray;'>Empowering minds through early detection and awareness</p>
    </div>
""", unsafe_allow_html=True)

# 🌈 Sidebar Inputs
st.sidebar.header("📝 Personal & Work Details")

def user_inputs():
    age = st.sidebar.slider("Age", 18, 65, 25)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    family_history = st.sidebar.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    work_interfere = st.sidebar.selectbox("Mental Health Interferes with Work?", ["Often", "Rarely", "Never", "Sometimes"])
    remote_work = st.sidebar.selectbox("Do You Work Remotely?", ["Yes", "No"])
    tech_company = st.sidebar.selectbox("Do You Work in Tech?", ["Yes", "No"])
    benefits = st.sidebar.selectbox("Does Your Employer Provide Mental Health Benefits?", ["Yes", "No", "Don't know"])
    care_options = st.sidebar.selectbox("Are You Aware of Care Options from Employer?", ["Yes", "No", "Not sure"])

    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'family_history': [family_history],
        'work_interfere': [work_interfere],
        'remote_work': [remote_work],
        'tech_company': [tech_company],
        'benefits': [benefits],
        'care_options': [care_options]
    })

# 📋 Get User Input
input_df = user_inputs()

# 🔁 Match Encoding with Training (same order, mapping)
label_encoders = {
    'Gender': {'Male': 1, 'Female': 0, 'Other': 2},
    'family_history': {'Yes': 1, 'No': 0},
    'work_interfere': {'Often': 3, 'Rarely': 2, 'Never': 0, 'Sometimes': 1},
    'remote_work': {'Yes': 1, 'No': 0},
    'tech_company': {'Yes': 1, 'No': 0},
    'benefits': {'Yes': 2, 'No': 0, "Don't know": 1},
    'care_options': {'Yes': 2, 'No': 0, 'Not sure': 1},
}

# 🧮 Apply encoding manually
for col, mapping in label_encoders.items():
    input_df[col] = input_df[col].map(mapping)

# ✨ Prediction
st.markdown("### 🧪 Prediction Result")
if st.button("Predict Mental Health Risk"):
    prediction = model.predict(input_df)[0]
    result = "🚨 Likely Needs Mental Health Treatment" if prediction == 1 else "✅ Unlikely to Need Immediate Treatment"
    
    st.success(result)

    if prediction == 1:
        st.markdown("""
        <div style='color: #c0392b; font-size: 16px;'>
            💡 Tip: Consider speaking to a mental health professional.<br>
            📞 Reach out to your HR or mental health helpline.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='color: #27ae60; font-size: 16px;'>
            🌿 Keep maintaining your mental well-being!<br>
            🧘 Try mindfulness, journaling, and regular breaks.
        </div>
        """, unsafe_allow_html=True)

# 📌 Footer
st.markdown("""
<hr style="border: 1px solid #ccc;">
<div style="text-align: center; font-size: 12px; color: gray;">
    Made with 💙 for awareness by WellMind AI.
</div>
""", unsafe_allow_html=True)
