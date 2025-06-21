import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('JOB_CHANGE.pkl')

st.title("Job Switch Prediction")

st.write("### Enter candidate information:")

commute_time = st.number_input("COMMUTE TIME? (note : in minutes)", min_value=0, max_value=300, value=30)
job_satisfaction = st.selectbox("JOB SATISFACTION LEVEL (1-EXCELLENT, 3-AVERAGE, 5-VERY BAD)", [1, 3, 5])
years_in_current_job = st.number_input("NUMBER OF YEARS IN CURRENT JOB?", min_value=0, max_value=50, value=2)
salary_expectation = st.number_input("SALARY EXPECTATION? (NOTE: LIKE THIS 20000)", min_value=0, max_value=1000000, value=20000)
wlb_input = st.selectbox("Work-Life Balance (WLB)", ["Yes", "No"])
wlb = 1 if wlb_input == "Yes" else 0

# Prepare input data (without feature names)
input_data = pd.DataFrame({
    "COMMUTE TIME? (note : in minutes)": [commute_time],
    "JOB SATISFACTION LEVEL (1-EXCELLENT,3-AVERAGE,5-VERY BAD)": [job_satisfaction],
    "NUMBER OF YEARS IN CURRENT JOB?": [years_in_current_job],
    "SALARY EXPECTATION? (NOTE: LIKE THIS 20000)": [salary_expectation],
    "WLB": [wlb]
})

# Convert dataframe to numpy array to remove feature names
input_array = input_data.to_numpy()

if st.button("Predict"):
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    if prediction[0] == 1:
        st.success("The candidate is likely to switch jobs.")
    else:
        st.info("The candidate is unlikely to switch jobs.")
        
    st.write("**Probability of switching:** {:.2f}%".format(prediction_proba[0][1]*100))
