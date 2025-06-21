import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('JOB_CHANGE.pkl')

# Streamlit app
st.title("Job Switch Prediction")

st.write("""
### Enter candidate details to predict job change probability:
""")

# Collect input features from user
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
relevant_experience = st.selectbox("Relevant Experience", ["Has Relevant Experience", "No Relevant Experience"])
enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course"])
education_level = st.selectbox("Education Level", ["Primary School", "High School", "Graduate", "Masters", "Phd"])
major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "Other", "No Major"])
experience = st.selectbox("Years of Experience", ['<1','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','>20'])
company_size = st.selectbox("Company Size", ["<10","10-49","50-99","100-500","500-999","1000-4999","5000-9999","10000+"])
company_type = st.selectbox("Company Type", ["Private", "Public", "Government", "Nonprofit", "Startup", "Other"])
last_new_job = st.selectbox("Last New Job (Years)", ["never", "1", "2", "3", "4", ">4"])
training_hours = st.number_input("Training Hours", min_value=0, max_value=500, value=20)

# You may need to preprocess these inputs exactly like your model expects.
# For now, let's assume you have a preprocessing pipeline inside the model or you manually do it here.

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'gender': [gender],
    'relevant_experience': [relevant_experience],
    'enrolled_university': [enrolled_university],
    'education_level': [education_level],
    'major_discipline': [major_discipline],
    'experience': [experience],
    'company_size': [company_size],
    'company_type': [company_type],
    'last_new_job': [last_new_job],
    'training_hours': [training_hours]
})

# Prediction button
if st.button('Predict'):
    # This assumes your model pipeline handles preprocessing
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("The candidate is likely to switch jobs.")
    else:
        st.info("The candidate is unlikely to switch jobs.")

    st.write("**Probability of switching:** {:.2f}%".format(prediction_proba[0][1] * 100))
