import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HeartCare AI - Smart Diagnosis",
    page_icon="❤️",
    layout="wide"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource # Isse baar-baar load nahi hoga (Performance boost)
def load_assets():
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR (Patient Info Summary) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("HeartCare AI")
    st.info("Ye model 300,000+ records par trained hai. Iska use sirf awareness ke liye karein.")
    st.divider()
    st.warning("⚠️ Disclaimer: Consult a doctor for medical advice.")

# --- MAIN PAGE LAYOUT ---
st.title("🏥 Smart Heart Disease Diagnostic Tool")
st.write("Fill the details below to analyze heart health using Machine Learning.")

# Form for better organization
with st.form("diagnosis_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧬 Physical Body")
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, help="Weight in kg / (height in m)^2")
        sex = st.selectbox("Biological Sex", ["Male", "Female"])
        age = st.select_slider("Age Category", options=range(0, 13), value=5, 
                               help="0=18-24, 12=80 or older")
        gen_health = st.select_slider("Overall Health Feel", options=range(0, 5), value=3,
                                      help="0: Poor, 4: Excellent")

    with col2:
        st.subheader("🚬 Lifestyle Factors")
        smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)
        alcohol = st.radio("Heavy Drinking?", ["Yes", "No"], horizontal=True)
        phys_act = st.radio("Active in last 30 days?", ["Yes", "No"], horizontal=True)
        sleep = st.number_input("Avg Sleep (Hours)", 0, 24, 7)

    with col3:
        st.subheader("📋 Medical History")
        stroke = st.checkbox("History of Stroke")
        asthma = st.checkbox("Asthma Patient")
        kidney = st.checkbox("Kidney Disease")
        skin_cancer = st.checkbox("Skin Cancer")
        diabetic = st.selectbox("Diabetes Status", ["No", "Yes", "No, borderline", "Yes (during pregnancy)"])

    # Submit button
    submitted = st.form_submit_button("GENERATE HEALTH REPORT")

# --- DATA PROCESSING ---
def get_clean_input():
    # Diabetic Status logic
    is_diabetic = 1 if diabetic == "Yes" else 0
    
    d = {
        'BMI': bmi,
        'Smoking': 1 if smoking == "Yes" else 0,
        'AlcoholDrinking': 1 if alcohol == "Yes" else 0,
        'Stroke': 1 if stroke else 0,
        'DiffWalking': 0,
        'Sex': 1 if sex == "Male" else 0,
        'AgeCategory': age,
        'PhysicalActivity': 1 if phys_act == "Yes" else 0,
        'GenHealth': gen_health,
        'SleepTime': sleep,
        'Asthma': 1 if asthma else 0,
        'KidneyDisease': 1 if kidney else 0,
        'SkinCancer': 1 if skin_cancer else 0,
        'Diabetic_Yes': is_diabetic,
        'Diabetic_No': 1 if diabetic == "No" else 0
    }
    
    input_df = pd.DataFrame([d])
    # Match columns with scaler
    for col in scaler.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[scaler.feature_names_in_]
    return scaler.transform(input_df)

# --- RESULTS ---
if submitted:
    input_data = get_clean_input()
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("### Result: POSITIVE")
            st.metric("Risk Level", "HIGH", delta="Action Required", delta_color="inverse")
        else:
            st.success("### Result: NEGATIVE")
            st.metric("Risk Level", "LOW", delta="Safe Range")

    with res_col2:
        st.write("### Analysis Breakdown")
        st.progress(probability)
        st.write(f"The model has calculated a **{probability:.1%}** probability of heart complications based on your inputs.")
        
        if probability > 0.5:
            st.warning("📣 Recommendation: Please schedule a checkup with a cardiologist as soon as possible.")
        else:
            st.info("📣 Recommendation: Your results look good! Continue maintaining a balanced diet and regular exercise.")