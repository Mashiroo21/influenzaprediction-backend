# streamlit_influenza_app.py
# Streamlit app with safer prediction handling when model expects more features than provided.
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date

st.set_page_config(page_title='Influenza Prediction', layout='centered')

# ---------- Load model (prefer pipeline) ----------
def load_model_prefer_pipeline(pipeline_path='model_pipeline.pkl', model_path='xgb_model.pkl'):
    """
    Try loading a full pipeline first (recommended). If not found, try loading a bare model.
    Returns (model_obj, is_pipeline_bool, load_path_str)
    """
    if os.path.exists(pipeline_path):
        m = joblib.load(pipeline_path)
        return m, True, pipeline_path
    if os.path.exists(model_path):
        m = joblib.load(model_path)
        return m, False, model_path
    return None, False, None

model, model_is_pipeline, model_path = load_model_prefer_pipeline()

if model is None:
    st.warning("Model not found: please place 'model_pipeline.pkl' (preferred) or 'xgb_model.pkl' in the app folder.")
    st.info("If you only have a raw model, save a scikit-learn Pipeline containing preprocessing + model (see hints below).")

# ---------- Feature builder from form payload ----------
def make_input(payload):
    # same minimal raw features you collect in forms (20)
    features = [
        'height','weight','temp','pulse','rr','sbp','o2s',
        'season','week_of_season','symptom_days','flu_vaccine',
        'expose_human','travel','cough','cough_with_sputum',
        'sore_throat','rhinorrhea','sinuspain','medhistav','pastmedchronlundis'
    ]
    row = []
    for f in features:
        v = payload.get(f)
        row.append(0 if v is None else v)
    return pd.DataFrame([row], columns=features)

# ---------- Helper to inspect model expected features ----------
def inspect_model_features(m):
    info = {}
    # sklearn estimators (XGBClassifier wrapper) often have n_features_in_
    info['type'] = type(m).__name__
    info['has_n_features_in_'] = hasattr(m, 'n_features_in_')
    info['n_features_in_'] = getattr(m, 'n_features_in_', None)
    # Pipeline?
    from sklearn.pipeline import Pipeline
    info['is_pipeline'] = isinstance(m, Pipeline)
    # If pipeline, try to inspect preprocessor/last step
    if info['is_pipeline']:
        try:
            # get final estimator
            final = m
            info['pipeline_steps'] = list(m.named_steps.keys())
            # attempt to get expected feature names from fitted ColumnTransformer / OneHotEncoder
            if hasattr(m, 'named_steps'):
                # find any transformer with get_feature_names_out
                fnames = {}
                for name, step in m.named_steps.items():
                    try:
                        if hasattr(step, 'get_feature_names_out'):
                            fnames[name] = step.get_feature_names_out()
                    except Exception:
                        pass
                info['pipeline_feature_extractors'] = list(fnames.keys())
        except Exception:
            pass
    # XGBoost Booster feature names
    try:
        if hasattr(m, 'get_booster'):
            booster = m.get_booster()
            info['booster_feature_names'] = booster.feature_names
        elif hasattr(m, 'feature_names'):
            info['booster_feature_names'] = getattr(m, 'feature_names')
        else:
            info['booster_feature_names'] = None
    except Exception:
        info['booster_feature_names'] = None
    return info

# ---------- Helper: try safe predict with automatic mapping/padding ----------
def safe_predict(m, is_pipeline, X_raw, force_zero_fill=False):
    """
    m: loaded model or pipeline
    is_pipeline: whether m is a pipeline (then just call m.predict(X_raw))
    X_raw: pandas DataFrame with raw features (shape (1,20))
    force_zero_fill: if True, attempt to map X_raw columns into model's expected feature names (if available)
                     and zero-fill the rest (unsafe).
    Returns (success_bool, result_or_error_message, debug_dict)
    """
    debug = {}
    # If pipeline, pass raw X to pipeline (it should handle preprocessing)
    if is_pipeline:
        try:
            pred = m.predict(X_raw)
            return True, pred, debug
        except Exception as e:
            return False, f"Pipeline prediction failed: {e}", debug

    # Not a pipeline: bare model expects already-preprocessed features
    # Try to detect expected n_features or feature names
    expected = getattr(m, 'n_features_in_', None)
    debug['n_features_in_'] = expected

    if expected is None:
        # try to inspect booster feature names (XGBoost)
        try:
            booster = m.get_booster()
            feat_names = booster.feature_names
            debug['booster_feature_names'] = feat_names
            expected = len(feat_names) if feat_names is not None else None
        except Exception:
            debug['booster_feature_names'] = None

    # If expected matches our X_raw, done
    if expected is not None and expected == X_raw.shape[1]:
        try:
            pred = m.predict(X_raw)
            return True, pred, debug
        except Exception as e:
            return False, f"Prediction failed even though shapes match: {e}", debug

    # Shapes mismatch
    debug['raw_shape'] = X_raw.shape
    debug['expected'] = expected

    # If we can get booster feature names, we can attempt mapping
    booster_feature_names = debug.get('booster_feature_names')
    model_feature_names = None
    if booster_feature_names:
        model_feature_names = list(booster_feature_names)

    # If no feature names & no expected -> cannot proceed
    if expected is None and not model_feature_names:
        return False, ("Model does not expose expected feature count or feature names. "
                       "Cannot safely prepare inputs. Please re-save a pipeline including preprocessing."), debug

    # If force_zero_fill requested, create zero-filled DataFrame with model_feature_names or expected count
    if force_zero_fill:
        if model_feature_names:
            target_cols = model_feature_names
            newX = pd.DataFrame([0]*len(target_cols)).T
            newX.columns = target_cols
            # try to map any matching columns from X_raw into newX
            for c in X_raw.columns:
                if c in newX.columns:
                    newX.at[0, c] = X_raw.at[0, c]
            try:
                pred = m.predict(newX)
                debug['used_zero_fill_with_names'] = True
                return True, pred, debug
            except Exception as e:
                return False, f"Prediction with zero-fill mapping failed: {e}", debug
        else:
            # no names, only expected count
            cols = [f'f{i}' for i in range(expected)]
            newX = pd.DataFrame([[0]*expected], columns=cols)
            # map first len(X_raw.columns) values into first columns (best-effort)
            for i, c in enumerate(X_raw.columns):
                if i < expected:
                    newX.iat[0, i] = X_raw.iat[0, i]
            try:
                pred = m.predict(newX)
                debug['used_zero_fill_by_index'] = True
                return True, pred, debug
            except Exception as e:
                return False, f"Prediction with zero-fill by index failed: {e}", debug

    # If reach here, return informative error and diagnostics
    msg = ("Feature shape mismatch: model expects %s features but input has %d. "
           "You can (A) provide a saved preprocessing pipeline (recommended), "
           "or (B) enable force-zero-fill to attempt prediction (unsafe).") % (str(expected), X_raw.shape[1])
    return False, msg, debug

# ---------- UI - collect forms (Home, Form1, Form2, Result) ----------
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'form1' not in st.session_state:
    st.session_state['form1'] = {}
if 'form2' not in st.session_state:
    st.session_state['form2'] = {}

def go_to(page):
    st.session_state['page'] = page

if st.session_state['page'] == 'Home':
    st.markdown("""
    <style>
    .top-background{height:220px;background:linear-gradient(90deg,#90AEFF,#0045FF);border-radius:12px}
    .card{background:white;padding:30px;border-radius:12px;margin-top:-100px;box-shadow:0 6px 20px rgba(0,0,0,0.08);}
    .title{font-size:22px;font-weight:700;margin-bottom:18px}
    </style>
    <div class="top-background"></div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Influenza Prediction</div>', unsafe_allow_html=True)
    if st.button('START'):
        go_to('FormPage1')
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state['page'] == 'FormPage1':
    st.markdown('# Please Fill in the Form')
    st.write('Please fill according to your actual condition')
    with st.form('form1_form'):
        height = st.text_input('Height (cm)', '')
        weight = st.text_input('Weight (kg)', '')
        temp = st.text_input('Body Temperature (°C)', '')
        pulse = st.text_input('Pulse (/min)', '')
        o2s = st.text_input('Oxygen Saturation (%)', '')
        rr = st.text_input('Respiration Rate (/min)', '')
        sbp = st.text_input('Systolic Blood Pressure (mmHg)', '')
        submitted = st.form_submit_button('Next')
    if submitted:
        st.session_state['form1'] = {
            'height': float(height) if height else 0,
            'weight': float(weight) if weight else 0,
            'temp': float(temp) if temp else 0,
            'pulse': float(pulse) if pulse else 0,
            'o2s': float(o2s) if o2s else 0,
            'rr': float(rr) if rr else 0,
            'sbp': float(sbp) if sbp else 0,
        }
        go_to('FormPage2')

elif st.session_state['page'] == 'FormPage2':
    st.markdown('# Additional Info')
    form1 = st.session_state.get('form1', {})
    with st.form('form2_form'):
        date_val = st.date_input('Date', date.today())
        season = st.text_input('Season (e.g., Winter)')
        flu_vaccine = st.radio('Did you ever get flu vaccine?', ('No','Yes'))
        expose_human = st.radio('Were you exposed to other sick people?', ('No','Yes'))
        travelled = st.radio('Did you go travelling in past 30 days?', ('No','Yes'))
        cough = st.radio('Did you have cough?', ('No','Yes'))
        cough_sputum = st.radio('Cough with sputum?', ('No','Yes'))
        sore_throat = st.radio('Did you have sore throat?', ('No','Yes'))
        rhinorrhea = st.radio('Do you have rhinorrhea?', ('No','Yes'))
        sinuspain = st.radio('Do you feel sinus pain?', ('No','Yes'))
        medhistav = st.radio('Do you have medhistav?', ('No','Yes'))
        pastmed = st.radio('Do you have past chronic lung disease?', ('No','Yes'))
        symptom_days = st.text_input('How long have the symptoms been present? (days)', '')
        submitted2 = st.form_submit_button('Next')
    if submitted2:
        start = date(date_val.year, 1, 1)
        week_of_season = ((date_val - start).days // 7) + 1
        st.session_state['form2'] = {
            'date': date_val.isoformat(),
            'season': season,
            'week_of_season': week_of_season,
            'symptom_days': int(symptom_days) if symptom_days else 0,
            'flu_vaccine': 1 if flu_vaccine == 'Yes' else 0,
            'expose_human': 1 if expose_human == 'Yes' else 0,
            'travel': 1 if travelled == 'Yes' else 0,
            'cough': 1 if cough == 'Yes' else 0,
            'cough_with_sputum': 1 if cough_sputum == 'Yes' else 0,
            'sore_throat': 1 if sore_throat == 'Yes' else 0,
            'rhinorrhea': 1 if rhinorrhea == 'Yes' else 0,
            'sinuspain': 1 if sinuspain == 'Yes' else 0,
            'medhistav': 1 if medhistav == 'Yes' else 0,
            'pastmedchronlundis': 1 if pastmed == 'Yes' else 0,
        }
        go_to('Result')

elif st.session_state['page'] == 'Result':
    st.markdown('# Your Result')
    form1 = st.session_state.get('form1', {})
    form2 = st.session_state.get('form2', {})

    if model is None:
        st.error("Model file not found. Place 'model_pipeline.pkl' (preferred) or 'xgb_model.pkl' in the app folder.")
        if st.button('Back to Home'):
            go_to('Home')
    else:
        payload = {}
        payload.update(form1)
        payload.update(form2)
        X = make_input(payload)  # DataFrame shape (1, 20)

        st.subheader("Debug / Model Info")
        info = inspect_model_features(model)
        st.write("Loaded from:", model_path)
        st.write("Model type:", info.get('type'))
        st.write("Is pipeline:", info.get('is_pipeline'))
        st.write("Model n_features_in_:", info.get('n_features_in_'))
        if info.get('booster_feature_names'):
            st.write("Booster feature names sample (first 20):", info.get('booster_feature_names')[:20])
        st.write("Your input columns:", list(X.columns))
        st.write("Your input shape:", X.shape)

        # Try normal prediction or safe predict
        force = st.checkbox("Force zero-fill and predict if shapes mismatch (unsafe; may be inaccurate)")
        success, result_or_msg, debug = safe_predict(model, model_is_pipeline, X, force_zero_fill=force)

        if success:
            pred = result_or_msg
            label = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
            infected = label == 1
            if infected:
                st.success("Result: Infected")
            else:
                st.success("Result: Not Infected")
            st.write("Raw model output:", pred.tolist() if hasattr(pred, 'tolist') else pred)
            st.write("Debug info:", debug)
            if st.button("Retry"):
                st.session_state['form1'] = {}
                st.session_state['form2'] = {}
                go_to('Home')
        else:
            # show helpful error + remediation steps
            st.error("Error during prediction:")
            st.write(result_or_msg)
            st.write("Debug info:", debug)
            st.markdown("**How to fix (recommended):**")
            st.markdown("""
            1. During training, build a sklearn `Pipeline` that includes your preprocessing (scaling, one-hot encoding, etc)
               and the final estimator. Then save the pipeline, e.g.:
            ```py
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from xgboost import XGBClassifier
            import joblib

            numeric_cols = ['height','weight','temp','pulse','rr','sbp','o2s','symptom_days']
            categorical_cols = ['season','...']

            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ])

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', XGBClassifier(...))
            ])

            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, 'model_pipeline.pkl')
            ```
            2. Place that `model_pipeline.pkl` in the app folder and restart Streamlit. The app will auto-load it (preferred).
            """)
            st.markdown("**Temporary workaround (unsafe):** enable the checkbox above to force zero-fill mapping, which will try to match your 20 inputs to the model's expected features and set the rest to zero. This can give unstable/incorrect predictions — use only to test integration.")
            if st.button("Back to Home"):
                go_to('Home')
