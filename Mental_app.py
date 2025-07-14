import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# 🌐 Set page configuration
st.set_page_config(page_title="WellMind - Mental Health Predictor", page_icon="🧠")

# Title and description
st.title("🧠 WellMind: Mental Health Risk Predictor")
st.markdown("""
Welcome to **WellMind**, an intelligent tool designed to assess the likelihood of needing mental health support based on your workplace environment and personal factors.
Please fill out the form below to get a prediction.
""")

# 🎛️ Input function
def get_user_input():
    age = st.slider("Age", 18, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does mental health affect your work?", ["Often", "Rarely", "Never", "Sometimes"])
    no_employees = st.selectbox("Number of employees", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Do you work in tech?", ["Yes", "No"])
    benefits = st.selectbox("Mental health benefits offered?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Are you aware of care options from employer?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness programs available?", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Is it easy to seek help?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of taking mental health leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Consequences of discussing mental health?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Consequences of discussing physical health?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfortable with coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfortable with supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Willing to discuss mental health in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Willing to discuss physical health in interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Mental health as important as physical?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed consequences of disclosure?", ["Yes", "No"])
    country = st.selectbox("Country", ["India", "United States", "Canada", "Other"])

    # Return as dataframe
    return pd.DataFrame({
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
        'Country': [country]
    })

# 🚀 Manual encoding to match training setup
def encode_input(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

# Collect user input
input_df = get_user_input()

# Encode it
encoded_input = encode_input(input_df)

# ✅ Match columns with training model's features
expected_features = model.feature_names_in_

# Add any missing columns as zero
for col in expected_features:
    if col not in encoded_input.columns:
        encoded_input[col] = 0

# Reorder columns to match model training
encoded_input = encoded_input[expected_features]

# 📊 Predict
if st.button("🔍 Predict Mental Health Risk"):
    prediction = model.predict(encoded_input)[0]
    if prediction == 1:
        st.error("🔴 You may benefit from mental health support.")
        st.markdown("💡 Consider speaking with a mental health professional or using employee support programs.")
    else:
        st.success("🟢 You are unlikely to need mental health intervention.")
        st.markdown("🎉 Keep maintaining a healthy lifestyle and positive mental wellness!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Made with 💙 by WellMind AI | Mental Health Awareness Tool")
