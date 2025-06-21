import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('JOB_CHANGE.pkl')

# Custom page config
st.set_page_config(page_title="Job Switch Prediction", page_icon="🧑‍💼", layout="centered")

# Custom CSS for animation and style
st.markdown("""
    <style>
    .title {
        font-size:40px;
        font-weight:bold;
        color:#2E86C1;
        text-align: center;
    }
    .subtitle {
        font-size:20px;
        color:#117A65;
        text-align: center;
    }
    .credits {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        padding: 15px 25px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 16px;
        animation: float 3s ease-in-out infinite;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    @keyframes float {
        0% { transform: translatey(0px); }
        50% { transform: translatey(-10px); }
        100% { transform: translatey(0px); }
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">Job Switch Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict if a candidate is likely to switch jobs based on key parameters</div>', unsafe_allow_html=True)
st.write("---")

# Input fields
st.write("### Please fill in candidate details:")

commute_time = st.number_input("🚗 Commute Time (in minutes)", min_value=0, max_value=300, value=30)
job_satisfaction = st.selectbox("😊 Job Satisfaction Level", [1, 3, 5], help="1=Excellent, 3=Average, 5=Very Bad")
years_in_current_job = st.number_input("📅 Number of Years in Current Job", min_value=0, max_value=50, value=2)
salary_expectation = st.number_input("💰 Salary Expectation (₹)", min_value=0, max_value=1000000, value=20000)
wlb_input = st.selectbox("⚖️ Work-Life Balance (WLB)", ["Yes", "No"])
wlb = 1 if wlb_input == "Yes" else 0

# Prepare input data
input_data = pd.DataFrame({
    "COMMUTE TIME? (note : in minutes)": [commute_time],
    "JOB SATISFACTION LEVEL (1-EXCELLENT,3-AVERAGE,5-VERY BAD)": [job_satisfaction],
    "NUMBER OF YEARS IN CURRENT JOB?": [years_in_current_job],
    "SALARY EXPECTATION? (NOTE: LIKE THIS 20000)": [salary_expectation],
    "WLB": [wlb]
})

input_array = input_data.to_numpy()

# Predict button
if st.button("🔮 Predict Job Switch"):
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    if prediction[0] == 1:
        st.success("✅ The candidate is likely to switch jobs.")
    else:
        st.info("❌ The candidate is unlikely to switch jobs.")
        
    st.write("**Probability of switching:** {:.2f}%".format(prediction_proba[0][1]*100))

# Floating animated credit bubble
st.markdown("""
    <div class="credits">
        PROJECT DONE BY Lognath, Thanmanan, Rithick
    </div>
""", unsafe_allow_html=True)
