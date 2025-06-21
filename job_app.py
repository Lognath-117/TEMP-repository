import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('JOB_CHANGE.pkl')

# Page config
st.set_page_config(page_title="Job Switch Prediction", page_icon="🧑‍💼", layout="centered")

# Background image insertion
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .title {
        font-size:45px;
        font-weight:bold;
        color:#ffffff;
        text-align: center;
        padding: 20px;
        text-shadow: 2px 2px 8px #000000;
    }
    .subtitle {
        font-size:20px;
        color:#F1C40F;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 5px #000000;
    }
    .input-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.4);
    }
    .banner {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 50px;
        background: linear-gradient(90deg, #ff0080, #7928ca);
        overflow: hidden;
        white-space: nowrap;
        z-index: 9999;
    }
    .banner-text {
        display: inline-block;
        font-weight: bold;
        font-size: 24px;
        color: white;
        padding-left: 100%;
        animation: marquee 15s linear infinite;
    }
    @keyframes marquee {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🚀 Job Switch Prediction System 🚀</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict if a candidate is likely to switch jobs based on real data</div>', unsafe_allow_html=True)

# Input container
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.write("### Enter Candidate Details:")

commute_time = st.number_input("🚗 Commute Time (in minutes)", min_value=0, max_value=300, value=30)
job_satisfaction = st.selectbox("😊 Job Satisfaction Level", [1, 3, 5], help="1=Excellent, 3=Average, 5=Very Bad")
years_in_current_job = st.number_input("📅 Number of Years in Current Job", min_value=0, max_value=50, value=2)
salary_expectation = st.number_input("💰 Salary Expectation (₹)", min_value=0, max_value=1000000, value=20000)
wlb_input = st.selectbox("⚖️ Work-Life Balance (WLB)", ["Yes", "No"])
wlb = 1 if wlb_input == "Yes" else 0
st.markdown('</div>', unsafe_allow_html=True)

# Prepare input
input_data = pd.DataFrame({
    "COMMUTE TIME? (note : in minutes)": [commute_time],
    "JOB SATISFACTION LEVEL (1-EXCELLENT,3-AVERAGE,5-VERY BAD)": [job_satisfaction],
    "NUMBER OF YEARS IN CURRENT JOB?": [years_in_current_job],
    "SALARY EXPECTATION? (NOTE: LIKE THIS 20000)": [salary_expectation],
    "WLB": [wlb]
})

input_array = input_data.to_numpy()

# Predict button
if st.button("🎯 Predict Job Switch"):
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    if prediction[0] == 1:
        st.success("✅ The candidate is likely to switch jobs.")
        st.balloons()
    else:
        st.info("❌ The candidate is unlikely to switch jobs.")
        st.snow()

    st.write("**Probability of switching:** {:.2f}%".format(prediction_proba[0][1]*100))

# Full screen running banner
st.markdown("""
    <div class="banner">
        <div class="banner-text">🚀 PROJECT DONE BY Lognath, Thanmanan, Rithick 🚀 PROJECT DONE BY Lognath, Thanmanan, Rithick 🚀</div>
    </div>
""", unsafe_allow_html=True)
