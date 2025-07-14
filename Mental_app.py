import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Set up Streamlit UI
st.set_page_config(page_title="WellMind", page_icon="🧠")
st.title("🧠 WellMind: Mental Health Risk Predictor")

st.markdown("### Please fill the details below to predict your mental health support need:")

# Full input list based on model training
def user_input():
    age = st.slider("Age", 18, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does mental health affect your work?", ["Often", "Rarely", "Never", "Sometimes"])
    no_employees = st.selectbox("Number of employees in your company", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Do you work in tech?", ["Yes", "No"])
    benefits = st.selectbox("Mental health benefits provided by employer?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Awareness of care options from employer?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness programs offered?", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Ease of seeking help for mental health?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Anonymity provided by employer?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of taking medical leave?", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Consequences of discussing mental health at work?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Consequences of discussing physical health?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfortable talking to coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfortable talking to supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Willing to discuss mental health in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Willing to discuss physical health in interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Is mental health as important as physical?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Seen negative consequences of mental health disclosure?", ["Yes", "No"])

    # Create DataFrame
    data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'self_employed': [self_employed],
        'family_history': [family_history],
        'work_interfere': [work_interfere],
        'no_employees': [no_employees],
        'remote_work': [remote_work],
        'tech_company': [tech_company],
        'benefits': [benefits],
        'care_options': [care_options],
        'wellness_program': [wellness_program],
        'seek_help': [seek_help],
        'anonymity': [anonymity],
        'leave': [leave],
        'mental_health_consequence': [mental_health_consequence],
        'phys_health_consequence': [phys_health_consequence],
        'coworkers': [coworkers],
        'supervisor': [supervisor],
        'mental_health_interview': [mental_health_interview],
        'phys_health_interview': [phys_health_interview],
        'mental_vs_physical': [mental_vs_physical],
        'obs_consequence': [obs_consequence],
        'Country': ["India"]  # Defaulted (can be optional input too)
    })

    return data

# Get input from user
input_df = user_input()

# Label encode same way as training
def manual_label_encoding(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

encoded_input = manual_label_encoding(input_df)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(encoded_input)[0]
    if prediction == 1:
        st.error("🔴 Prediction: You may need mental health support.")
        st.markdown("💡 It's okay to ask for help. Consider consulting a mental health professional or counselor.")
    else:
        st.success("🟢 Prediction: You are unlikely to need immediate support.")
        st.markdown("✅ Keep maintaining your mental well-being!")

# Footer
st.markdown("<hr style='margin-top:30px;'>", unsafe_allow_html=True)
st.caption("🧠 Powered by WellMind | Mental Health Awareness Initiative")
