import streamlit as st
import pandas as pd
import joblib
import os
from groq import Groq

# ======================================
# CONFIG
# ======================================
st.set_page_config(
    page_title="Patient Dropout Prediction System",
    page_icon="🏥",
    layout="centered"
)

# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_model():
    return joblib.load("rf_patient_dropout_model.pkl")

rf_model = load_model()

# ======================================
# GROQ CLIENT
# ======================================
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
client = Groq()

# ======================================
# UI HEADER
# ======================================
st.title("🏥 Patient Dropout Prediction System")
st.subheader("AI-powered Appointment Attendance Prediction")

st.markdown("---")

# ======================================
# INPUT FORM
# ======================================
with st.form("patient_form"):
    st.markdown("### 🧑 Patient Information")

    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 0, 120, 30)
    scholarship = st.selectbox("Scholarship", ["No", "Yes"])
    hipertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
    handcap = st.selectbox("Handicap", ["No", "Yes"])
    sms_received = st.selectbox("SMS Received", ["No", "Yes"])
    waiting_days = st.slider("Waiting Days", 0, 365, 5)

    submitted = st.form_submit_button("🔍 Predict Dropout Risk")

# ======================================
# PREDICTION + LLM
# ======================================
if submitted:
    input_df = pd.DataFrame([{
        "gender": 1 if gender == "Male" else 0,
        "age": age,
        "scholarship": 1 if scholarship == "Yes" else 0,
        "hipertension": 1 if hipertension == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "alcoholism": 1 if alcoholism == "Yes" else 0,
        "handcap": 1 if handcap == "Yes" else 0,
        "sms_received": 1 if sms_received == "Yes" else 0,
        "waiting_days": waiting_days
    }])

    # ML Prediction
    dropout_prob = rf_model.predict_proba(input_df)[0][1]
    risk = "High Risk" if dropout_prob >= 0.5 else "Low Risk"

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    if risk == "High Risk":
        st.error(f"🚨 High Dropout Risk ({dropout_prob:.2f})")
    else:
        st.success(f"✅ Low Dropout Risk ({dropout_prob:.2f})")

    # ======================================
    # GROQ LLM RESPONSE
    # ======================================
    prompt = f"""
You are a healthcare analytics AI assistant.

Patient details:
Age: {age}
Gender: {gender}
Waiting days: {waiting_days}
SMS received: {sms_received}

ML Prediction:
Dropout Risk: {risk}
Probability: {dropout_prob:.2f}

Tasks:
1. Explain why the patient is at this risk.
2. Suggest hospital actions to reduce dropout.
3. Keep it professional and concise.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful healthcare AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    st.markdown("### 🧠 AI Explanation & Recommendations")
    st.write(response.choices[0].message.content)
