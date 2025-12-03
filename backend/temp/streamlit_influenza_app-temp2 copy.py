# streamlit_influenza_app.py
import streamlit as st
import pandas as pd
import joblib
import os
import pathlib
from datetime import datetime, date

# ---------- Helper: load model ----------
@st.cache_resource
def load_model(path="/backend/"):
    path = pathlib.Path(__file__).parent / path
    if not path.exists():
        st.error(f"Model not found at {path}")
        return None
    return joblib.load(path)

model = load_model("model_pipeline.pkl")

# ---------- UI Config ----------
st.set_page_config(page_title="Influenza Prediction", layout="centered")

PAGES = ["Home", "FormPage1", "FormPage2", "Result"]
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page

# ---------- Global CSS Styling ----------
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
    background-color: #f0f2f6;
}

/* ===== General Layout ===== */
.container {
    position: relative;
    height: 100vh;
    width: 100%;
    overflow: hidden;
}

.topBackground {
    position: absolute;
    top: 0;
    height: 45%;
    width: 100%;
    background: linear-gradient(180deg, #90AEFF, #0045FF);
}

.bottomBackground {
    position: absolute;
    bottom: 0;
    height: 55%;
    width: 100%;
    background-color: #f0f0f0;
}

.card {
    position: relative;
    background-color: #fff;
    width: 85%;
    min-height: 65%;
    padding: 30px;
    border-radius: 20px;
    margin: auto;
    margin-top: 60%;
    text-align: center;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

/* ===== Title & Header ===== */
.title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 25px;
}

/* ===== Buttons ===== */
div.stButton > button {
    width: 100%;
    border-radius: 25px;
    padding: 14px 30px;
    background: linear-gradient(90deg, #00FF9D, #4CF925);
    color: white;
    font-family: Inter, sans-serif;
    font-weight: bold;
    font-size: 14px;
    text-align: center;
    cursor: pointer;
    border: none;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: scale(1.05);
    opacity: 0.9;
}

/* ===== Form Fields ===== */
div[data-baseweb="input"] > div {
    border-radius: 25px !important;
    border: 1.5px solid #ccc !important;
    padding: 4px 12px !important;
}

div[data-baseweb="input"] input {
    border-radius: 25px;
    text-align: left;
}

div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 10px;
    padding: 6px 14px;
    border: 1px solid #ccc;
    border-radius: 25px;
    margin-bottom: 8px;
    transition: 0.2s;
}
div[role="radiogroup"] > label:hover {
    background-color: #f5f5f5;
}

/* ===== Submit Buttons (Next, Detail, Retry) ===== */
button[kind="formSubmit"] {
    background: linear-gradient(90deg, #0066FF, #0045FF) !important;
    color: white !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
}

/* ===== Result Circle ===== */
.result-circle {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    border: 8px solid #00FF9D;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 30px auto;
    font-size: 22px;
    font-weight: 700;
    color: #00FF9D;
    text-align: center;
}

/* ===== Result Buttons ===== */
.result-btn-green > button {
    width: 100%;
    background: linear-gradient(90deg, #00FF9D, #4CF925);
    color: white;
    font-weight: bold;
    border-radius: 25px;
    margin-top: 10px;
    border: none;
}
.result-btn-blue > button {
    width: 100%;
    background: linear-gradient(90deg, #0066FF, #0045FF);
    color: white;
    font-weight: bold;
    border-radius: 25px;
    margin-top: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------- Home ----------
if st.session_state.page == "Home":
    st.markdown(
        """
        <div class="container">
            <div class="topBackground"></div>
            <div class="bottomBackground"></div>
            <div class="card">
                <div class="title">Influenza Prediction</div>
        """,
        unsafe_allow_html=True
    )

    st.button("START", on_click=lambda: go_to("FormPage1"))

    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------- Form Page 1 ----------
elif st.session_state.page == "FormPage1":
    st.markdown("<h3 style='text-align:center;font-weight:700;'>Please Fill In the Form</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:13px;color:#777;'>Please fill in the form according to your actual condition</p>", unsafe_allow_html=True)

    with st.form("form1_ui"):
        height = st.text_input("height", "")
        weight = st.text_input("weight", "")
        temp = st.text_input("temp", "")
        pulse = st.text_input("pulse", "")
        o2s = st.text_input("o2s", "")
        rr = st.text_input("rr", "")
        sbp = st.text_input("sbp", "")
        st.markdown("<div style='text-align:right;font-size:12px;color:#888;'>Page 1/2</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Next")

    if submitted:
        st.session_state["form1"] = {
            "heightcm": float(height) if height else 0,
            "weightkg": float(weight) if weight else 0,
            "as_edenroll_temp": float(temp) if temp else 0,
            "pulse": float(pulse) if pulse else 0,
            "rr": float(rr) if rr else 0,
            "sbp": float(sbp) if sbp else 0,
            "o2s": float(o2s) if o2s else 0,
        }
        go_to("FormPage2")

# ---------- Form Page 2 ----------
elif st.session_state.page == "FormPage2":
    st.markdown("<h3 style='text-align:center;font-weight:700;'>Please Fill In the Form</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:13px;color:#777;'>Please fill in the form according to your actual condition</p>", unsafe_allow_html=True)

    with st.form("form2_ui"):
        date_val = st.date_input("Date", date.today())
        season = st.text_input("Season (e.g., Winter)")
        flu_vaccine = st.radio("Did you ever get flu vaccine?", ("No", "Yes"))
        expose_human = st.radio("Were you exposed to other sick people?", ("No", "Yes"))
        travelled = st.radio("Did you go travelling in past 30 days?", ("No", "Yes"))
        cough = st.radio("Did you have cough?", ("No", "Yes"))
        cough_sputum = st.radio("Cough with sputum?", ("No", "Yes"))
        sore_throat = st.radio("Did you have sore throat?", ("No", "Yes"))
        rhinorrhea = st.radio("Do you have rhinorrhea?", ("No", "Yes"))
        sinuspain = st.radio("Do you feel sinus pain?", ("No", "Yes"))
        medhistav = st.radio("Do you have medhistav?", ("No", "Yes"))
        pastmed = st.radio("Do you have past chronic lung disease?", ("No", "Yes"))
        symptom_days = st.text_input("How long have the symptoms been present? (days)", "")
        st.markdown("<div style='text-align:right;font-size:12px;color:#888;'>Page 2/2</div>", unsafe_allow_html=True)
        submitted2 = st.form_submit_button("Next")

    if submitted2:
        start = date(date_val.year, 1, 1)
        week_of_season = ((date_val - start).days // 7) + 1

        st.session_state["form2"] = {
            "season": int(season),
            "WOS": int(week_of_season),
            "cursympt_days": int(symptom_days) if symptom_days else 0,
            "fluvaccine": 1 if flu_vaccine == "Yes" else 0,
            "exposehuman": 1 if expose_human == "Yes" else 0,
            "travel": 1 if travelled == "Yes" else 0,
            "cursympt_cough": 1 if cough == "Yes" else 0,
            "cursympt_coughsputum": 1 if cough_sputum == "Yes" else 0,
            "cursympt_sorethroat": 1 if sore_throat == "Yes" else 0,
            "cursympt_rhinorrhea": 1 if rhinorrhea == "Yes" else 0,
            "cursympt_sinuspain": 1 if sinuspain == "Yes" else 0,
            "medhistav": 1 if medhistav == "Yes" else 0,
            "pastmedchronlundis": 1 if pastmed == "Yes" else 0,
        }
        go_to("Result")

# ---------- Result ----------
elif st.session_state.page == "Result":
    st.markdown("<h3 style='text-align:center;font-weight:700;'>Your Result</h3>", unsafe_allow_html=True)

    form1 = st.session_state.get("form1", {})
    form2 = st.session_state.get("form2", {})

    if model is None:
        st.error("Model file 'model_pipeline.pkl' not found. Please retrain and save.")
        if st.button("Back to Home"):
            go_to("Home")
    else:
        payload = {}
        payload.update(form1)
        payload.update(form2)
        try:
            X = pd.DataFrame([payload])
            pred = model.predict(X)
            pred_label = int(pred[0])
            infected = pred_label == 1
            st.markdown('<div class="" style="font-weight: bold;">Probability of Infection:</div>', unsafe_allow_html=True)
            if infected:
                st.markdown('<div class="result-circle" style="border-color:#FF4B4B;color:#FF4B4B;">Infected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-circle">Not Infected</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="result-btn-green">', unsafe_allow_html=True)
                st.button("Detail")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="result-btn-blue">', unsafe_allow_html=True)
                if st.button("Retry"):
                    st.session_state["form1"] = {}
                    st.session_state["form2"] = {}
                    go_to("Home")
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            if st.button("Back to Home"):
                go_to("Home")
