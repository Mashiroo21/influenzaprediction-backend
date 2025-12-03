import streamlit as st
import pandas as pd
import joblib
import os
import pathlib
from datetime import datetime, date

# ---------- Helper: load model ----------
@st.cache_resource
def load_model(path="model_pipeline.pkl"):
    path = pathlib.Path(__file__).parent / path
    if not path.exists():
        return None
    return joblib.load(path)

model = load_model("model_pipeline.pkl")

# ---------- Helper: Recommendation Logic ----------
def get_recommendations(data, prediction_label):
    recs_keys = set()
    recommendations = []

    # --- KATEGORI 1: RED FLAGS ---
    if float(data.get("o2s", 100)) < 95:
        recs_keys.add("O2S_LOW")
        recommendations.append(("Immediate Medical Attention", "Saturasi oksigen < 95%. Tanda hipoksemia serius.", "WHO", "danger"))
    if float(data.get("rr", 20)) > 24:
        recs_keys.add("RR_HIGH")
        recommendations.append(("Immediate Medical Attention", "Laju napas > 24x/menit. Distres pernapasan.", "Merck Manual", "danger"))
    if float(data.get("as_edenroll_temp", 36)) > 40:
        recs_keys.add("TEMP_EXTREME")
        recommendations.append(("Immediate Medical Attention", "Suhu > 40°C. Hiperpireksia.", "CDC", "danger"))
    if float(data.get("sbp", 120)) < 90:
        recs_keys.add("SBP_LOW")
        recommendations.append(("Immediate Medical Attention", "Tekanan darah sistolik < 90. Tanda syok.", "WHO", "danger"))

    if any(r[3] == 'danger' for r in recommendations):
        return recommendations

    # --- KATEGORI 2: RISIKO TINGGI ---
    temp = float(data.get("as_edenroll_temp", 36))
    if temp > 38.0:
        msg = "Demam > 38°C menandakan infeksi. Konsultasi dokter disarankan."
        recommendations.append(("Consult a Doctor", msg, "Panduan Medis Umum", "warning"))
    if data.get("pastmedchronlundis", 0) == 1:
        recommendations.append(("High Risk Factor", "Riwayat penyakit paru kronis meningkatkan risiko komplikasi.", "CDC", "warning"))

    # --- KATEGORI 3 & 4: EDUKASI ---
    if prediction_label == 1 and not recommendations:
         recommendations.append(("Self-Care", "Prediksi positif gejala ringan. Istirahat & hidrasi.", "CDC", "info"))
    elif prediction_label == 0 and not recommendations:
         recommendations.append(("General Advice", "Prediksi negatif. Kemungkinan common cold. Istirahat.", "CDC", "info"))
    
    if data.get("fluvaccine", 1) == 0:
        recommendations.append(("Prevention", "Pertimbangkan vaksin flu tahunan.", "WHO", "info"))

    return recommendations

# ---------- UI Config ----------
st.set_page_config(page_title="Influenza Prediction", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- NAVIGASI LOGIC ---
def go_to(page):
    st.session_state.page = page

def go_home():
    st.session_state["form1"] = {}
    st.session_state["form2"] = {}
    st.session_state.page = "Home"

# ---------- CSS MANAGEMENT ----------
ROSE_COLOR = "#E06377" 
BORDER_COLOR = "#E06377"
BTN_BLUE = "linear-gradient(90deg, #4B90FF, #0055FF)"
BTN_GREEN = "linear-gradient(90deg, #00FF9D, #4CF925)"

def load_css(page_type="form"):
    base_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root { --font: "Inter", sans-serif; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    [data-testid="stHeader"] { visibility: hidden; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if page_type == "home":
        # CSS HOME
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(180deg, #4B90FF 40%, #f0f2f6 40%);
        }}
        .home-card {{
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 85%; max-width: 400px; height: 50vh;
            background: white; border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            z-index: 1;
        }}
        .home-title {{
            font-size: 26px; font-weight: 800; color: #333; margin-bottom: 60px;
        }}
        div.stButton > button {{
            background: {BTN_GREEN} !important;
            color: white !important; font-weight: 700; border: none;
            border-radius: 30px; padding: 15px 40px;
            box-shadow: 0 10px 20px rgba(0,255,157, 0.4);
            position: fixed; top: 60%; left: 50%;
            transform: translate(-50%, -50%);
            z-index: 10; width: auto;
        }}
        div.stButton > button:hover {{ transform: translate(-50%, -50%) scale(1.05); }}
        </style>
        """, unsafe_allow_html=True)

    else:
        # CSS FORM & RESULT & DETAIL
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{ background-color: #f0f2f6; }}
        
        /* Container Card Putih */
        [data-testid="stAppViewContainer"] > .main > .block-container {{
            background-color: white;
            border-radius: 24px;
            padding: 3rem 2rem !important;
            margin-top: 30px; margin-bottom: 30px;
            max-width: 550px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }}

        /* Rose Theme Inputs */
        .stTextInput label, .stNumberInput label, .stDateInput label, .stSelectbox label, .stRadio label p {{
            color: {ROSE_COLOR} !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }}
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{
            border-radius: 20px !important;
            border: 1.5px solid {BORDER_COLOR} !important;
            background-color: white !important;
            color: #333 !important;
        }}
        input {{ color: #333 !important; }}
        
        /* Tombol Standard (Biru) */
        div.stButton > button {{
            background: {BTN_BLUE} !important;
            color: white !important; font-weight: 700; border: none;
            border-radius: 30px; padding: 12px 0;
            width: 100%;
            box-shadow: 0 5px 15px rgba(75, 144, 255, 0.3);
        }}
        div.stButton > button:hover {{ transform: scale(1.02); }}
        
        .page-indicator {{ color: {ROSE_COLOR}; font-size: 12px; font-weight: 500; margin-top: 15px; }}
        .form-title {{ text-align: center; font-weight: 800; font-size: 22px; color: #333; }}
        .form-subtitle {{ text-align: center; font-size: 12px; color: #888; margin-bottom: 25px; }}
        </style>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE: HOME
# ==========================================
if st.session_state.page == "Home":
    load_css("home")
    st.markdown('<div class="home-card"><div class="home-title">Influenza Prediction</div></div>', unsafe_allow_html=True)
    st.button("START", on_click=lambda: go_to("FormPage1"))


# ==========================================
# PAGE: FORM 1
# ==========================================
elif st.session_state.page == "FormPage1":
    load_css("form") 
    st.markdown('<div class="form-title">Please Fill In the Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">Please Fill In the Form according to your actual condition</div>', unsafe_allow_html=True)
    
    with st.form("form1_ui"):
        height = st.text_input("Height", "")
        weight = st.text_input("Weight", "")
        temp = st.text_input("Temperature", "")
        pulse = st.text_input("Pulse", "")
        o2s = st.text_input("Oxygen Saturation", "")
        rr = st.text_input("Respiratory Rate", "")
        sbp = st.text_input("Systolic Blood Pressure", "")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
             st.markdown('<div class="page-indicator">Page 1/2</div>', unsafe_allow_html=True)
        with c2:
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
        st.session_state.page = "FormPage2"
        st.rerun()


# ==========================================
# PAGE: FORM 2
# ==========================================
elif st.session_state.page == "FormPage2":
    load_css("form")
    st.markdown('<div class="form-title">Please Fill In the Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">Please Fill In the Form according to your actual condition</div>', unsafe_allow_html=True)

    with st.form("form2_ui"):
        date_val = st.date_input("Date", date.today())
        season = st.selectbox("Season (e.g. 1=Spring)", ["1", "2", "3", "4"]) 
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        flu_vaccine = st.radio("Did you ever get flu vaccine?", ("No", "Yes"))
        travelled = st.radio("Did you go travelling in past 30 days?", ("No", "Yes"))
        expose_human = st.radio("Were you exposed to other sick people?", ("No", "Yes"))
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        cough = st.radio("Did you have cough?", ("No", "Yes"))
        sore_throat = st.radio("Did you have sore throat?", ("No", "Yes"))
        cough_sputum = st.radio("Cough with sputum?", ("No", "Yes"))
        rhinorrhea = st.radio("Do you have rhinorrhea?", ("No", "Yes"))
        sinuspain = st.radio("Do you feel sinus pain?", ("No", "Yes"))
        medhistav = st.radio("Medical History Available?", ("No", "Yes"))
        pastmed = st.radio("Past chronic lung disease?", ("No", "Yes"))
        symptom_days = st.number_input("How long have the symptoms been present? (days)", min_value=0, value=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
             st.markdown('<div class="page-indicator">Page 2/2</div>', unsafe_allow_html=True)
        with c2:
             submitted2 = st.form_submit_button("Next")

    if submitted2:
        start = date(date_val.year, 1, 1)
        week_of_season = ((date_val - start).days // 7) + 1
        st.session_state["form2"] = {
            "season": int(season) if season else 1,
            "WOS": int(week_of_season),
            "cursympt_days": int(symptom_days),
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
        st.session_state.page = "Result"
        st.rerun()


# ==========================================
# PAGE: RESULT
# ==========================================
elif st.session_state.page == "Result":
    load_css("form")
    st.markdown('<h3 style="text-align:center;font-weight:700; color:#333;">Prediction Result</h3>', unsafe_allow_html=True)

    form1 = st.session_state.get("form1", {})
    form2 = st.session_state.get("form2", {})

    if model is None:
        st.error("Model not found.")
        st.button("Home", on_click=go_home)
    else:
        payload = {}
        payload.update(form1)
        payload.update(form2)
        try:
            X = pd.DataFrame([payload])
            pred = model.predict(X)
            pred_label = int(pred[0])
            st.session_state["last_pred_label"] = pred_label
            infected = pred_label == 1
            
            if infected:
                st.markdown("""
                <div style="width:180px; height:180px; border-radius:50%; border:8px solid #FF4B4B; 
                color:#FF4B4B; display:flex; align-items:center; justify-content:center; 
                margin:20px auto; font-size:24px; font-weight:bold;">Infected</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="width:180px; height:180px; border-radius:50%; border:8px solid #00FF9D; 
                color:#00C853; display:flex; align-items:center; justify-content:center; 
                margin:20px auto; font-size:24px; font-weight:bold;">Not Infected</div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<style>div.stButton button {{ background: {BTN_GREEN} !important; }}</style>""", unsafe_allow_html=True)
                st.button("Detail", key="btn_detail", on_click=lambda: go_to("DetailPage"))
            with col2:
                st.markdown(f"""<style>div.stButton button {{ background: {BTN_BLUE} !important; }}</style>""", unsafe_allow_html=True)
                st.button("Retry", key="btn_retry", on_click=go_home)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.button("Home", on_click=go_home)


# ==========================================
# PAGE: DETAIL
# ==========================================
elif st.session_state.page == "DetailPage":
    load_css("form") # Base container style

    # --- CSS KHUSUS DETAIL PAGE ---
    st.markdown("""
    <style>
    /* Styling Disclaimer Box */
    .disclaimer-box {
        background-color: #FFF8E1;
        border: 1px solid #FFE0B2;
        color: #E65100;
        padding: 15px;
        border-radius: 12px;
        font-size: 13px;
        margin-bottom: 25px;
        line-height: 1.5;
    }
    
    /* Styling Recommendation Card */
    .rec-card {
        background-color: #FFFFFF;
        border: 1px solid #F0F0F0;
        border-left-width: 6px; /* Border warna di kiri */
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: transform 0.2s;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(0,0,0,0.08);
    }
    
    /* Variant Colors */
    .rec-danger { border-left-color: #FF4B4B; }
    .rec-warning { border-left-color: #FFB020; }
    .rec-info { border-left-color: #4B90FF; }
    
    /* Typography dalam Card */
    .rec-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .rec-icon { font-size: 18px; }
    .rec-title { font-weight: 700; font-size: 15px; color: #333; margin: 0; }
    .rec-body { font-size: 13px; color: #555; line-height: 1.5; margin-bottom: 8px; }
    .rec-source { font-size: 11px; color: #999; font-style: italic; text-align: right; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h3 style="text-align:center; font-weight:700; color:#333; margin-bottom:20px;">Medical Advice</h3>', unsafe_allow_html=True)
    
    # Disclaimer Box
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ PENTING:</strong> Hasil prediksi ini <strong>bukanlah diagnosis medis</strong>. 
        Aplikasi ini hanya bersifat prediktif dan edukatif. Untuk diagnosis dan perawatan yang akurat, 
        harap segera konsultasikan dengan dokter.
    </div>
    """, unsafe_allow_html=True)
    
    form1 = st.session_state.get("form1", {})
    form2 = st.session_state.get("form2", {})
    pred_label = st.session_state.get("last_pred_label", 0)
    
    all_data = {}
    all_data.update(form1)
    all_data.update(form2)
    
    recs = get_recommendations(all_data, pred_label)
    
    # Render Cards
    if not recs:
        st.info("Tidak ada rekomendasi khusus. Tetap jaga kesehatan!")
    
    for title, text, src, level in recs:
        # Tentukan Icon berdasarkan level
        icon = "🚨" if level == "danger" else "⚠️" if level == "warning" else "ℹ️"
        
        st.markdown(f"""
        <div class="rec-card rec-{level}">
            <div class="rec-header">
                <span class="rec-icon">{icon}</span>
                <span class="rec-title">{title}</span>
            </div>
            <div class="rec-body">{text}</div>
            <div class="rec-source">Source: {src}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Navigation Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<style>div.stButton button {{ background: {BTN_BLUE} !important; }}</style>""", unsafe_allow_html=True)
        st.button("Back", key="btn_back", on_click=lambda: go_to("Result"))
    with col2:
        st.markdown(f"""<style>div.stButton button {{ background: {BTN_BLUE} !important; }}</style>""", unsafe_allow_html=True)
        st.button("Home", key="btn_home", on_click=go_home)