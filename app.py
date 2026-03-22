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

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Make sure these filenames match your GitHub files exactly
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}. Check if files exist in repo.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("HeartCare AI")
    st.info("Trained on 300,000+ records. Use for awareness only.")
    st.divider()
    st.warning("⚠️ Disclaimer: Consult a doctor for medical advice.")

# --- MAIN PAGE ---
st.title("🏥 Smart Heart Disease Diagnostic Tool")
st.write("Fill the details below to analyze heart health using Machine Learning.")

# --- AGE CATEGORY MAPPING (Fixing the 12 limit issue) ---
age_labels = [
    "18-24", "25-29", "30-34", "35-39", "40-44", 
    "45-49", "50-54", "55-59", "60-64", "65-69", 
    "70-74", "75-79", "80 or older"
]

with st.form("diagnosis_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧬 Physical Body")
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
        sex = st.selectbox("Biological Sex", ["Male", "Female"])
        
        # FIX: Ab user ko range dikhegi (0-12 ke bajaye years dikhenge)
        selected_age = st.select_slider("Age Category", options=age_labels, value="40-44")
        # Background mein index (0-12) nikalna
        age_index = age_labels.index(selected_age)
        
        gen_health = st.select_slider("Overall Health Feel", options=["Poor", "Fair", "Good", "Very Good", "Excellent"], value="Good")
        # Convert health to numbers (0-4)
        health_map = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4}
        gen_health_num = health_map[gen_health]

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

    submitted = st.form_submit_button("GENERATE HEALTH REPORT")

# --- DATA PROCESSING ---
def get_clean_input():
    is_diabetic = 1 if diabetic == "Yes" else 0
    
    d = {
        'BMI': bmi,
        'Smoking': 1 if smoking == "Yes" else 0,
        'AlcoholDrinking': 1 if alcohol == "Yes" else 0,
        'Stroke': 1 if stroke else 0,
        'DiffWalking': 0, # Default value
        'Sex': 1 if sex == "Male" else 0,
        'AgeCategory': age_index, # Fixed index
        'PhysicalActivity': 1 if phys_act == "Yes" else 0,
        'GenHealth': gen_health_num,
        'SleepTime': sleep,
        'Asthma': 1 if asthma else 0,
        'KidneyDisease': 1 if kidney else 0,
        'SkinCancer': 1 if skin_cancer else 0,
        'Diabetic_Yes': is_diabetic,
        'Diabetic_No': 1 if diabetic == "No" else 0
    }
    
    input_df = pd.DataFrame([d])
    
    # Ensuring column order matches scaler
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
        if prediction == 1 or probability > 0.5:
            st.error("### Result: POSITIVE")
            st.metric("Risk Level", "HIGH", delta="Action Required", delta_color="inverse")
        else:
            st.success("### Result: NEGATIVE")
            st.metric("Risk Level", "LOW", delta="Safe Range")

    with res_col2:
        st.write("### Analysis Breakdown")
        st.progress(probability)
        st.write(f"The model has calculated a **{probability:.1%}** probability of heart complications.")
        
        if probability > 0.5:
            st.warning("📣 Recommendation: Please schedule a checkup with a cardiologist.")
        else:
            st.info("📣 Recommendation: Maintain a balanced diet and regular exercise.")
