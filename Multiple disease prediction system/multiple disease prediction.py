import pickle
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Assistant | Multiple Disease Prediction",
    page_icon="🩺",
    layout="wide",
)

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
    }
    .hero h1, .hero p { color: #ffffff !important; margin: 0; }
    .hero h1 { font-size: 2rem; font-weight: 700; padding: 0 0 0.3rem 0; }
    .hero p  { font-size: 1rem; opacity: 0.92; }

    .section-title {
        font-size: 1.05rem; font-weight: 600; margin: 0.6rem 0 0.2rem 0;
        padding-bottom: 0.3rem; border-bottom: 2px solid rgba(37, 99, 235, 0.35);
    }

    .result-card {
        padding: 1.2rem 1.5rem; border-radius: 14px; margin-top: 1.2rem;
        border-left: 7px solid; font-size: 1.1rem;
    }
    .result-card h3 { margin: 0 0 0.3rem 0; font-size: 1.25rem; }
    .result-card p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }
    .result-pos { background: rgba(220, 38, 38, 0.12); border-color: #dc2626; }
    .result-neg { background: rgba(22, 163, 74, 0.12); border-color: #16a34a; }

    .disclaimer {
        margin-top: 2rem; padding: 0.9rem 1.2rem; border-radius: 12px;
        background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.45);
        font-size: 0.88rem;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        color: #ffffff; border: none; padding: 0.65rem 2rem;
        border-radius: 10px; font-weight: 600; font-size: 1rem;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.08); color: #ffffff; border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load models (cached so they load only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    folder = BASE / "saved models"
    with open(folder / "diabetes_model.sav", "rb") as f:
        diabetes = pickle.load(f)
    with open(folder / "heart_disease_model.sav", "rb") as f:
        heart = pickle.load(f)
    with open(folder / "parkinsons_model.sav", "rb") as f:
        parkinsons = pickle.load(f)
    return diabetes, heart, parkinsons


diabetes_model, heart_disease_model, parkinsons_model = load_models()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def show_result(positive: bool, positive_text: str, negative_text: str):
    if positive:
        st.markdown(
            f'<div class="result-card result-pos"><h3>⚠️ {positive_text}</h3>'
            "<p>The model indicates a higher likelihood. Please consult a "
            "qualified doctor for a proper diagnosis.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card result-neg"><h3>✅ {negative_text}</h3>'
            "<p>The model indicates a lower likelihood. Regular health "
            "check-ups are still recommended.</p></div>",
            unsafe_allow_html=True,
        )


def disclaimer():
    st.markdown(
        '<div class="disclaimer"><b>Disclaimer:</b> This tool is built for '
        "educational purposes using machine learning models. It is not a "
        "medical device and must not be used as a substitute for professional "
        "medical advice, diagnosis or treatment.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "Health Assistant",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["droplet-half", "heart-pulse", "soundwave"],
        menu_icon="hospital-fill",
        default_index=0,
        styles={
            "container": {"padding": "0.5rem", "background-color": "transparent"},
            "menu-title": {"font-weight": "700", "font-size": "1.2rem"},
            "nav-link": {"font-size": "0.95rem", "border-radius": "10px", "margin": "4px 0"},
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #0f766e 0%, #2563eb 100%)",
                "color": "white",
            },
        },
    )
    st.caption("Built with Python, scikit-learn and Streamlit.")


# ---------------------------------------------------------------------------
# Diabetes prediction
# ---------------------------------------------------------------------------
if selected == "Diabetes Prediction":
    hero(
        "🩸 Diabetes Prediction",
        "Enter the patient's health measurements to estimate the likelihood of diabetes.",
    )

    with st.form("diabetes_form"):
        section("Patient details")
        c1, c2, c3 = st.columns(3)
        with c1:
            pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0, step=1)
            skin = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f")
        with c2:
            glucose = st.number_input("Glucose level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
            insulin = st.number_input("Insulin level (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        with c3:
            bp = st.number_input("Blood pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")

        submitted = st.form_submit_button("🔍 Get Diabetes Test Result")

    if submitted:
        features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        prediction = diabetes_model.predict([[float(x) for x in features]])
        show_result(
            prediction[0] == 1,
            "The person is likely diabetic",
            "The person is likely not diabetic",
        )

    disclaimer()


# ---------------------------------------------------------------------------
# Heart disease prediction
# ---------------------------------------------------------------------------
if selected == "Heart Disease Prediction":
    hero(
        "❤️ Heart Disease Prediction",
        "Provide clinical details to estimate the likelihood of heart disease.",
    )

    # Label -> numeric code used by the trained model
    SEX = {"Male": 1, "Female": 0}
    CHEST_PAIN = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3,
    }
    FASTING_BS = {"No (≤ 120 mg/dL)": 0, "Yes (> 120 mg/dL)": 1}
    REST_ECG = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2,
    }
    EXERCISE_ANGINA = {"No": 0, "Yes": 1}
    ST_SLOPE = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    THAL = {"Normal": 0, "Fixed defect": 1, "Reversible defect": 2}

    with st.form("heart_form"):
        section("Personal information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        with c2:
            sex = st.selectbox("Sex", list(SEX.keys()))
        with c3:
            cp = st.selectbox("Chest pain type", list(CHEST_PAIN.keys()))

        section("Vitals and lab results")
        c1, c2, c3 = st.columns(3)
        with c1:
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
            restecg = st.selectbox("Resting ECG result", list(REST_ECG.keys()))
        with c2:
            chol = st.number_input("Serum cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)
            thalach = st.number_input("Maximum heart rate achieved (bpm)", min_value=60, max_value=220, value=150, step=1)
        with c3:
            fbs = st.selectbox("Fasting blood sugar", list(FASTING_BS.keys()))
            exang = st.selectbox("Exercise-induced angina", list(EXERCISE_ANGINA.keys()))

        section("Exercise test and imaging")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
        with c2:
            slope = st.selectbox("Slope of peak exercise ST segment", list(ST_SLOPE.keys()))
        with c3:
            ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
        with c4:
            thal = st.selectbox("Thalassemia", list(THAL.keys()))

        submitted = st.form_submit_button("🔍 Get Heart Disease Test Result")

    if submitted:
        features = [
            age,
            SEX[sex],
            CHEST_PAIN[cp],
            trestbps,
            chol,
            FASTING_BS[fbs],
            REST_ECG[restecg],
            thalach,
            EXERCISE_ANGINA[exang],
            oldpeak,
            ST_SLOPE[slope],
            ca,
            THAL[thal],
        ]
        prediction = heart_disease_model.predict([[float(x) for x in features]])
        show_result(
            prediction[0] == 1,
            "The person is likely to have heart disease",
            "The person is unlikely to have heart disease",
        )

    disclaimer()


# ---------------------------------------------------------------------------
# Parkinson's prediction
# ---------------------------------------------------------------------------
if selected == "Parkinsons Prediction":
    hero(
        "🧠 Parkinson's Disease Prediction",
        "Enter voice-measurement values to estimate the likelihood of Parkinson's disease.",
    )

    # Order of features across the groups matches the order used to train the model
    PARKINSON_GROUPS = {
        "🎵 Frequency": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
        "📈 Jitter": [
            "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)",
            "MDVP:RAP",
            "MDVP:PPQ",
            "Jitter:DDP",
        ],
        "📊 Shimmer": [
            "MDVP:Shimmer",
            "MDVP:Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "MDVP:APQ",
            "Shimmer:DDA",
        ],
        "🔬 Noise and nonlinear measures": [
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "spread1",
            "spread2",
            "D2",
            "PPE",
        ],
    }
    FEATURE_ORDER = [name for names in PARKINSON_GROUPS.values() for name in names]

    st.caption("Fill in each tab. Values come from a sustained-vowel voice recording analysis.")

    with st.form("parkinsons_form"):
        values = {}
        tabs = st.tabs(list(PARKINSON_GROUPS.keys()))
        for tab, names in zip(tabs, PARKINSON_GROUPS.values()):
            with tab:
                cols = st.columns(3)
                for i, name in enumerate(names):
                    with cols[i % 3]:
                        values[name] = st.number_input(
                            name,
                            value=0.0,
                            step=0.0001,
                            format="%.5f",
                            key=f"pd_{name}",
                        )

        submitted = st.form_submit_button("🔍 Get Parkinson's Test Result")

    if submitted:
        features = [float(values[name]) for name in FEATURE_ORDER]
        prediction = parkinsons_model.predict([features])
        show_result(
            prediction[0] == 1,
            "The person is likely to have Parkinson's disease",
            "The person is unlikely to have Parkinson's disease",
        )

    disclaimer()
