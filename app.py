import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide"
)

# ================= LOAD MODEL =================
if not os.path.exists("svc.pkl") or not os.path.exists("features.pkl"):
    st.error("❌ Model not found. Train first.")
    st.stop()

svc = pickle.load(open("svc.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ================= LOAD DATA =================
description = pd.read_csv("description.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")
workout = pd.read_csv("workout_df.csv")

# ================= DISEASE MAPPING =================
diseases_list = {
15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue',
37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C',
21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack',
39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# ================= FUNCTIONS =================
def predict_disease(symptoms):
    input_df = pd.DataFrame(0, index=[0], columns=features)
    for symptom in symptoms:
        if symptom in input_df.columns:
            input_df[symptom] = 1
    return svc.predict(input_df)[0]


def helper(dis):
    dis = dis.strip()

    desc = description[description['Disease'] == dis]['Description']
    desc = desc.values[0] if len(desc) else "No description available"

    pre = precautions[precautions['Disease'] == dis].iloc[:, 1:].values
    med = medications[medications['Disease'] == dis]['Medication'].values
    die = diets[diets['Disease'] == dis]['Diet'].values
    wrkout = workout[workout['disease'].str.strip() == dis]['workout'].values

    return desc, pre, med, die, wrkout

# ================= UI =================

st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🩺 MediAssist AI</h1>
<p style='text-align: center; font-size:18px;'>Select your symptoms and get AI-based health insights</p>
""", unsafe_allow_html=True)

st.divider()

# INPUT
selected_symptoms = st.multiselect(
    "🔍 Select Symptoms",
    features,
    placeholder="Start typing symptoms..."
)

# BUTTON
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🚀 Predict Disease")

# ================= OUTPUT =================

if predict_btn:

    if selected_symptoms:

        disease_encoded = predict_disease(selected_symptoms)
        label = diseases_list[disease_encoded]

        desc, pre, med, die, wrkout = helper(label)

        st.divider()

        # RESULT
        st.success(f"🧠 **Predicted Disease: {label}**")

        # ================= RISK SCORE =================
        st.subheader("📊 Diagnostic Probability Index")

        risk_score = (len(selected_symptoms) / len(features)) * 100

        st.metric(label="Risk Score", value=f"{risk_score:.2f}%")
        st.progress(int(risk_score))

        if risk_score < 20:
            st.success("🟢 Low risk detected .")
        elif risk_score < 50:
            st.warning("🟡 Moderate risk detected .")
        else:
            st.error("🔴 High risk — don't ignore symptoms.")


        # ================= COLUMNS =================
        col1, col2 = st.columns(2)

        # LEFT
        with col1:
            st.subheader("📖 Description")
            st.info(desc)

            st.subheader("⚠️ Precautions")
            if len(pre) > 0:
                for i, p in enumerate(pre[0], 1):
                    st.write(f"{i}. {p}")
            else:
                st.write("No precautions available")

            st.subheader("🏃 Lifestyle")
            if len(wrkout) > 0:
                for i, w in enumerate(wrkout, 1):
                    st.write(f"{i}. {w}")
            else:
                st.write("No recommendations")

        # RIGHT
        with col2:
            st.subheader("💊 Medications")
            if len(med) > 0:
                for i, m in enumerate(med, 1):
                    st.write(f"{i}. {m}")
            else:
                st.write("No medications available")

            st.subheader("🥗 Diet Plan")
            if len(die) > 0:
                for i, d in enumerate(die, 1):
                    st.write(f"{i}. {d}")
            else:
                st.write("No diet suggestions")

    else:
        st.warning("⚠️ Please select at least one symptom")

# FOOTER
st.divider()
st.markdown(
    "<p style='text-align:center;'>⚡ Built with AI + Streamlit | For educational purposes only</p>",
    unsafe_allow_html=True
)