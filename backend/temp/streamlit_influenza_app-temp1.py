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

# ---------- UI Layout ----------
st.set_page_config(page_title="Influenza Prediction", layout="centered")

PAGES = ["Home", "FormPage1", "FormPage2", "Result"]
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page

# ---------- Home ----------
if st.session_state.page == "Home":
    st.markdown(
        """
        <style>
        .container {
            position: relative;
            height: 100vh;
            width: 100%;
            overflow: hidden;
        }

        .topBackground {
            position: absolute;
            top: 0;
            height: 40%;
            width: 100%;
            background: linear-gradient(180deg, #90AEFF, #0045FF);
        }

        .bottomBackground {
            position: absolute;
            bottom: 0;
            height: 60%;
            width: 100%;
            background-color: #f0f0f0;
        }

        .card {
            position: relative;
            background-color: #fff;
            width: 85%;
            height: 65%;
            padding: 30px;
            border-radius: 12px;
            margin: auto;
            margin-top: 60%;
            text-align: center;
            box-shadow: 0px 6px 8px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Custom style for streamlit button */
        div.stButton > button {
            width: 100%;
            border-radius: 25px;
            padding: 15px 30px;
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
        </style>
        """,
        unsafe_allow_html=True
    )

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

    if st.button("START"):
        go_to("FormPage1")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # st.markdown("""
    # <style>
    # .top-background{height:220px;background:linear-gradient(90deg,#90AEFF,#0045FF);border-radius:12px}
    # .card{background:white;padding:30px;border-radius:12px;margin-top:-100px;box-shadow:0 6px 20px rgba(0,0,0,0.08);}
    # .title{font-size:22px;font-weight:700;margin-bottom:18px}
    # </style>
    # <div class="top-background"></div>
    # """, unsafe_allow_html=True)

    # st.markdown('<div class="card">', unsafe_allow_html=True)
    # st.markdown('<div class="title">Influenza Prediction</div>', unsafe_allow_html=True)
    # if st.button("START"):
    #     go_to("FormPage1")
    # st.markdown("</div>", unsafe_allow_html=True)

# ---------- Form Page 1 ----------
elif st.session_state.page == "FormPage1":
    st.header("Please Fill in the Form")
    st.write("Please fill according to your actual condition")

    with st.form("form1_ui"):  # gunakan key unik
        height = st.text_input("height", "")
        weight = st.text_input("weight", "")
        temp = st.text_input("temp", "")
        pulse = st.text_input("pulse", "")
        o2s = st.text_input("o2s", "")
        rr = st.text_input("rr", "")
        sbp = st.text_input("sbp", "")
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
    st.header("Additional Info")
    form1 = st.session_state.get("form1", {})

    with st.form("form2_ui"):  # gunakan key unik
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
        submitted2 = st.form_submit_button("Next")

    if submitted2:
        start = date(date_val.year, 1, 1)  # pastikan keduanya bertipe date
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
    st.header("Your Result")
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

            if infected:
                st.markdown("<div style='text-align:center'><h2 style='color:red'>Infected</h2></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:center'><h2 style='color:green'>Not Infected</h2></div>", unsafe_allow_html=True)

            # st.write("Prediction raw output: ", pred.tolist())

            if st.button("Retry"):
                st.session_state["form1"] = {}
                st.session_state["form2"] = {}
                go_to("Home")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            if st.button("Back to Home"):
                go_to("Home")
