import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
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
IST = timezone(timedelta(hours=5, minutes=30))

# Optional: fill these in to show them on the About page
DEVELOPER_NAME = ""
GITHUB_URL = ""

# ---------------------------------------------------------------------------
# Styling (works in light and dark themes)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        padding: 1.8rem 2.2rem; border-radius: 18px; margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
    }
    .hero h1, .hero p { color: #ffffff !important; margin: 0; }
    .hero h1 { font-size: 2rem; font-weight: 700; padding: 0 0 0.3rem 0; }
    .hero p  { font-size: 1rem; opacity: 0.92; }

    .section-title {
        font-size: 1.05rem; font-weight: 600; margin: 0.6rem 0 0.2rem 0;
        padding-bottom: 0.3rem; border-bottom: 2px solid rgba(37, 99, 235, 0.35);
    }

    .card {
        padding: 1.2rem 1.3rem; border-radius: 16px; height: 100%;
        background: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.25);
    }
    .card h3 { margin: 0 0 0.4rem 0; font-size: 1.15rem; }
    .card p  { margin: 0.2rem 0; font-size: 0.93rem; opacity: 0.88; }
    .card .tag {
        display: inline-block; margin-top: 0.5rem; padding: 0.15rem 0.6rem;
        border-radius: 999px; font-size: 0.75rem; font-weight: 600;
        background: rgba(37, 99, 235, 0.15); color: #2563eb;
    }
    .big-icon { font-size: 2rem; }

    .helpline {
        padding: 1rem 1.2rem; border-radius: 14px; margin-bottom: 0.9rem;
        background: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.25);
    }
    .helpline .num a {
        font-size: 1.5rem; font-weight: 800; text-decoration: none; color: #dc2626;
    }
    .helpline .name { font-weight: 600; }
    .helpline .desc { font-size: 0.88rem; opacity: 0.85; }

    .emergency {
        padding: 1rem 1.4rem; border-radius: 14px; margin-bottom: 1.2rem;
        background: rgba(220, 38, 38, 0.12); border: 1px solid rgba(220, 38, 38, 0.5);
    }
    .emergency a { color: #dc2626; font-weight: 800; text-decoration: none; }

    .tip-box {
        padding: 1rem 1.4rem; border-radius: 14px; margin: 1.2rem 0;
        background: rgba(22, 163, 74, 0.12); border: 1px solid rgba(22, 163, 74, 0.45);
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
# Session state (lives only in the user's browser session, nothing is stored)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "results" not in st.session_state:
    st.session_state.results = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_str():
    return datetime.now(IST).strftime("%d %b %Y, %I:%M %p")


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def card(icon, title, lines, tag=""):
    body = "".join(f"<p>{line}</p>" for line in lines)
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="card"><div class="big-icon">{icon}</div>'
        f"<h3>{title}</h3>{body}{tag_html}</div>",
        unsafe_allow_html=True,
    )


def emergency_banner():
    st.markdown(
        '<div class="emergency">🚨 <b>In an emergency</b> (severe chest pain, trouble '
        "breathing, fainting, sudden weakness or confusion) do not rely on this app. "
        'Call <a href="tel:112">112</a> or ambulance <a href="tel:108">108</a> '
        "immediately.</div>",
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


def show_result(positive, positive_text, negative_text):
    if positive:
        st.markdown(
            f'<div class="result-card result-pos"><h3>⚠️ {positive_text}</h3>'
            "<p>The model indicates a higher likelihood. Please consult a "
            "qualified doctor for a proper diagnosis. See the Doctor Helpline "
            "page for the right specialist.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card result-neg"><h3>✅ {negative_text}</h3>'
            "<p>The model indicates a lower likelihood. Regular health "
            "check-ups are still recommended.</p></div>",
            unsafe_allow_html=True,
        )


def save_result(kind, positive, inputs, ranges=None):
    record = {
        "time": now_str(),
        "patient": st.session_state.get("patient_name", "").strip() or "Anonymous",
        "test": kind,
        "positive": bool(positive),
        "result": "Higher likelihood detected" if positive else "Lower likelihood detected",
        "inputs": inputs,
        "ranges": ranges or [],
    }
    st.session_state.results[kind] = record
    st.session_state.history.append(record)


def build_report(rec):
    lines = [
        "MULTIPLE DISEASE PREDICTION SYSTEM - SCREENING REPORT",
        "=" * 54,
        f"Patient : {rec['patient']}",
        f"Date    : {rec['time']}",
        f"Test    : {rec['test']}",
        f"Result  : {rec['result']}",
        "",
        "Input values:",
    ]
    for key, value in rec["inputs"].items():
        lines.append(f"  - {key}: {value}")
    if rec.get("ranges"):
        lines += ["", "Reference range check:"]
        for name, value, low, high, ref in rec["ranges"]:
            lines.append(f"  - {name}: {value} (reference {ref})")
    lines += [
        "",
        "NOTE: This is an educational machine learning screening, not a medical",
        "diagnosis. Please consult a qualified doctor for any health concern.",
    ]
    return "\n".join(lines)


def range_table(rows):
    data = []
    for name, value, low, high, ref in rows:
        status = "✅ Within range"
        if low is not None and value < low:
            status = "⚠️ Below range"
        elif high is not None and value > high:
            status = "⚠️ Above range"
        data.append(
            {"Parameter": name, "Your value": str(value), "Reference": ref, "Status": status}
        )
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)


def render_saved(kind, positive_text, negative_text):
    rec = st.session_state.results.get(kind)
    if not rec:
        return
    show_result(rec["positive"], positive_text, negative_text)
    st.caption(f"Report generated on {rec['time']} for {rec['patient']}")
    if rec.get("ranges"):
        section("Your values vs general reference ranges")
        range_table(rec["ranges"])
        st.caption("General adult reference ranges for information only. Your doctor may use different targets.")
    st.download_button(
        "⬇️ Download report",
        build_report(rec),
        file_name=f"{kind.lower().replace(' ', '_')}_report.txt",
        mime="text/plain",
        key=f"dl_{kind}",
    )


DAILY_TIPS = [
    "Walk for at least 30 minutes a day. Even short walks after meals help control blood sugar.",
    "Cut back on added sugar and ultra-processed snacks. Fruit, nuts and curd make better swaps.",
    "Limit salt. Too much sodium is a major driver of high blood pressure.",
    "Aim for 7 to 8 hours of sleep. Poor sleep raises the risk of heart disease and diabetes.",
    "Drink water regularly through the day and keep sugary drinks occasional.",
    "Get your blood pressure, sugar and cholesterol checked at least once a year after age 30.",
    "Add a handful of vegetables to every meal. Fibre slows sugar absorption.",
    "Quit smoking. It is one of the biggest single steps you can take for your heart.",
    "Manage stress with breathing exercises, hobbies or time outdoors.",
    "Strength exercises twice a week protect muscles, bones and metabolism.",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = [
    "Home",
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Parkinsons Prediction",
    "Health Calculators",
    "Doctor Helpline",
    "Health Library",
    "My Reports",
    "About",
]

with st.sidebar:
    selected = option_menu(
        "Health Assistant",
        PAGES,
        icons=[
            "house",
            "droplet-half",
            "heart-pulse",
            "soundwave",
            "calculator",
            "telephone",
            "book",
            "file-earmark-text",
            "info-circle",
        ],
        menu_icon="hospital-fill",
        default_index=0,
        styles={
            "container": {"padding": "0.5rem", "background-color": "transparent"},
            "menu-title": {"font-weight": "700", "font-size": "1.2rem"},
            "nav-link": {"font-size": "0.92rem", "border-radius": "10px", "margin": "3px 0"},
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #0f766e 0%, #2563eb 100%)",
                "color": "white",
            },
        },
    )
    st.text_input("Patient name (optional)", key="patient_name", placeholder="Used on your report")
    st.markdown(
        '<div class="emergency" style="margin-top:0.8rem;font-size:0.9rem;">'
        '🚨 Emergency?<br>Call <a href="tel:112">112</a> or <a href="tel:108">108</a></div>',
        unsafe_allow_html=True,
    )
    st.caption("Built with Python, scikit-learn and Streamlit. Your inputs are not stored on any server.")


# ===========================================================================
# HOME
# ===========================================================================
if selected == "Home":
    hero(
        "🩺 Multiple Disease Prediction System",
        "AI-powered early screening for Diabetes, Heart Disease and Parkinson's Disease, "
        "with health calculators, doctor guidance and emergency helplines in one place.",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Disease models", "3")
    m2.metric("Clinical parameters", "43")
    m3.metric("ML algorithms", "SVM + Logistic Regression")
    m4.metric("Screenings this session", len(st.session_state.history))

    emergency_banner()

    section("Choose a screening")
    c1, c2, c3 = st.columns(3)
    with c1:
        card(
            "🩸",
            "Diabetes",
            [
                "Estimate diabetes risk from 8 health measurements such as glucose, BMI and age.",
            ],
            "SVM · Pima Indians dataset",
        )
    with c2:
        card(
            "❤️",
            "Heart Disease",
            [
                "Assess heart disease likelihood from 13 clinical indicators including ECG and chest pain type.",
            ],
            "Logistic Regression · UCI dataset",
        )
    with c3:
        card(
            "🧠",
            "Parkinson's Disease",
            [
                "Analyse 22 voice-recording measures to screen for Parkinson's disease.",
            ],
            "SVM · UCI voice dataset",
        )
    st.caption("Use the menu on the left to open a screening.")

    section("How it works")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        card("1️⃣", "Enter details", ["Fill in your measurements using simple, guided inputs."])
    with s2:
        card("2️⃣", "Model analyses", ["A trained machine learning model evaluates your values instantly."])
    with s3:
        card("3️⃣", "Review result", ["See the outcome along with your values compared to reference ranges."])
    with s4:
        card("4️⃣", "Take action", ["Download your report and find the right doctor or helpline."])

    section("More tools")
    t1, t2, t3 = st.columns(3)
    with t1:
        card("🧮", "Health Calculators", ["BMI, daily calories, blood pressure and blood sugar checkers."])
    with t2:
        card("📞", "Doctor Helpline", ["Emergency numbers, telemedicine and a which-doctor-to-see guide."])
    with t3:
        card("📚", "Health Library", ["Prevention tips, warning signs and answers to common questions."])

    tip = DAILY_TIPS[datetime.now(IST).date().toordinal() % len(DAILY_TIPS)]
    st.markdown(
        f'<div class="tip-box">💡 <b>Health tip of the day</b><br>{tip}</div>',
        unsafe_allow_html=True,
    )
    disclaimer()


# ===========================================================================
# DIABETES
# ===========================================================================
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
            glucose = st.number_input("Glucose level (mg/dL)", min_value=1.0, max_value=300.0, value=120.0, step=1.0)
            insulin = st.number_input("Insulin level (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        with c3:
            bp = st.number_input("Blood pressure, diastolic (mm Hg)", min_value=1.0, max_value=200.0, value=70.0, step=1.0)
            bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")

        submitted = st.form_submit_button("🔍 Get Diabetes Test Result")

    if submitted:
        features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        prediction = diabetes_model.predict([[float(x) for x in features]])
        save_result(
            "Diabetes",
            prediction[0] == 1,
            {
                "Pregnancies": pregnancies,
                "Glucose (mg/dL)": glucose,
                "Blood pressure (mm Hg)": bp,
                "Skin thickness (mm)": skin,
                "Insulin (mu U/mL)": insulin,
                "BMI": bmi,
                "Diabetes pedigree function": dpf,
                "Age": age,
            },
            ranges=[
                ("Glucose (mg/dL)", glucose, 70, 99, "70 to 99 (fasting)"),
                ("Blood pressure, diastolic", bp, 60, 80, "60 to 80"),
                ("BMI", bmi, 18.5, 24.9, "18.5 to 24.9"),
            ],
        )

    render_saved("Diabetes", "The person is likely diabetic", "The person is likely not diabetic")
    disclaimer()


# ===========================================================================
# HEART DISEASE
# ===========================================================================
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
        save_result(
            "Heart Disease",
            prediction[0] == 1,
            {
                "Age": age,
                "Sex": sex,
                "Chest pain type": cp,
                "Resting BP (mm Hg)": trestbps,
                "Cholesterol (mg/dL)": chol,
                "Fasting blood sugar": fbs,
                "Resting ECG": restecg,
                "Max heart rate (bpm)": thalach,
                "Exercise-induced angina": exang,
                "ST depression": oldpeak,
                "ST slope": slope,
                "Major vessels": ca,
                "Thalassemia": thal,
            },
            ranges=[
                ("Resting blood pressure (mm Hg)", trestbps, 90, 120, "90 to 120"),
                ("Cholesterol (mg/dL)", chol, None, 200, "below 200"),
                ("Max heart rate (bpm)", thalach, None, 220 - age, f"up to {220 - age} (age-predicted max)"),
            ],
        )

    render_saved(
        "Heart Disease",
        "The person is likely to have heart disease",
        "The person is unlikely to have heart disease",
    )
    disclaimer()


# ===========================================================================
# PARKINSON'S
# ===========================================================================
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

    st.caption("Fill in each tab. Values come from the analysis of a sustained-vowel voice recording.")

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
        if all(float(v) == 0 for v in values.values()):
            st.warning("Please enter the voice measurements before running the test.")
        else:
            features = [float(values[name]) for name in FEATURE_ORDER]
            prediction = parkinsons_model.predict([features])
            save_result("Parkinsons", prediction[0] == 1, {k: values[k] for k in FEATURE_ORDER})

    render_saved(
        "Parkinsons",
        "The person is likely to have Parkinson's disease",
        "The person is unlikely to have Parkinson's disease",
    )
    disclaimer()


# ===========================================================================
# HEALTH CALCULATORS
# ===========================================================================
if selected == "Health Calculators":
    hero(
        "🧮 Health Calculators",
        "Quick tools to understand your BMI, calorie needs, blood pressure and blood sugar.",
    )

    tab_bmi, tab_bp, tab_sugar = st.tabs(["⚖️ BMI and Calories", "🩺 Blood Pressure", "🍬 Blood Sugar"])

    with tab_bmi:
        c1, c2, c3 = st.columns(3)
        with c1:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=65.0, step=0.5)
            sex_c = st.selectbox("Sex", ["Male", "Female"], key="calc_sex")
        with c2:
            height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=0.5)
            activity = st.selectbox(
                "Activity level",
                [
                    "Sedentary (little or no exercise)",
                    "Light (1 to 3 days a week)",
                    "Moderate (3 to 5 days a week)",
                    "Active (6 to 7 days a week)",
                    "Very active (hard exercise daily)",
                ],
            )
        with c3:
            age_c = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1, key="calc_age")

        h_m = height / 100
        bmi_val = weight / (h_m**2)
        if bmi_val < 18.5:
            bmi_cat, bmi_color = "Underweight", "🟡"
        elif bmi_val < 25:
            bmi_cat, bmi_color = "Normal weight", "🟢"
        elif bmi_val < 30:
            bmi_cat, bmi_color = "Overweight", "🟠"
        else:
            bmi_cat, bmi_color = "Obese", "🔴"

        multipliers = [1.2, 1.375, 1.55, 1.725, 1.9]
        levels = [
            "Sedentary (little or no exercise)",
            "Light (1 to 3 days a week)",
            "Moderate (3 to 5 days a week)",
            "Active (6 to 7 days a week)",
            "Very active (hard exercise daily)",
        ]
        bmr = 10 * weight + 6.25 * height - 5 * age_c + (5 if sex_c == "Male" else -161)
        tdee = bmr * multipliers[levels.index(activity)]
        low_w, high_w = 18.5 * h_m**2, 24.9 * h_m**2
        max_hr = 220 - age_c

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BMI", f"{bmi_val:.1f}", f"{bmi_color} {bmi_cat}", delta_color="off")
        m2.metric("Healthy weight range", f"{low_w:.0f} to {high_w:.0f} kg")
        m3.metric("Daily calories (maintain)", f"{tdee:,.0f} kcal")
        m4.metric("Daily water", f"{weight * 0.035:.1f} litres")
        st.progress(min(max((bmi_val - 10) / 30, 0.0), 1.0), text=f"BMI position: {bmi_val:.1f} (scale 10 to 40)")
        st.info(
            f"Estimated resting energy use (BMR): about {bmr:,.0f} kcal/day. "
            f"Target heart-rate zone for moderate exercise: {int(max_hr * 0.5)} to {int(max_hr * 0.7)} bpm "
            f"(estimated max {max_hr} bpm)."
        )
        st.caption("BMI does not account for muscle mass or body fat. Calorie and water values are general estimates.")

    with tab_bp:
        c1, c2 = st.columns(2)
        with c1:
            sys_bp = st.number_input("Systolic (top number, mm Hg)", min_value=60, max_value=260, value=118, step=1)
        with c2:
            dia_bp = st.number_input("Diastolic (bottom number, mm Hg)", min_value=30, max_value=160, value=76, step=1)

        if sys_bp > 180 or dia_bp > 120:
            st.error("🚨 Hypertensive crisis range. If you also have chest pain, breathlessness, weakness or vision changes, call 112 now. Otherwise recheck in 5 minutes and contact a doctor urgently.")
        elif sys_bp >= 140 or dia_bp >= 90:
            st.error("🔴 Stage 2 hypertension range. Please consult a doctor soon.")
        elif sys_bp >= 130 or dia_bp >= 80:
            st.warning("🟠 Stage 1 hypertension range. Lifestyle changes and a doctor's review are advised.")
        elif sys_bp >= 120:
            st.warning("🟡 Elevated. Watch salt intake, stay active and recheck regularly.")
        else:
            st.success("🟢 Normal blood pressure.")
        st.caption("Based on common adult guideline categories. A diagnosis needs repeated readings measured by a professional.")

    with tab_sugar:
        c1, c2 = st.columns(2)
        with c1:
            sugar_type = st.selectbox("Type of reading", ["Fasting (8+ hours without food)", "2 hours after a meal"])
        with c2:
            sugar_val = st.number_input("Blood glucose (mg/dL)", min_value=20, max_value=600, value=95, step=1)

        if sugar_type.startswith("Fasting"):
            low_limit, high_limit = 100, 126
        else:
            low_limit, high_limit = 140, 200

        if sugar_val < 70:
            st.error("🔴 Low blood sugar. If you feel shaky, sweaty or confused, take fast-acting sugar (juice or glucose) and seek medical help.")
        elif sugar_val < low_limit:
            st.success("🟢 Normal range.")
        elif sugar_val < high_limit:
            st.warning("🟠 Prediabetes range. Lifestyle changes and a doctor's advice are recommended.")
        else:
            st.error("🔴 In the diabetes range. This needs confirmation by a doctor with a lab test such as HbA1c.")
        st.caption("A single reading cannot diagnose diabetes. Confirm with a doctor.")

    disclaimer()


# ===========================================================================
# DOCTOR HELPLINE
# ===========================================================================
if selected == "Doctor Helpline":
    hero(
        "📞 Doctor Helpline and Emergency Support",
        "Important numbers, free telemedicine and a guide to the right specialist.",
    )
    emergency_banner()

    section("Helpline numbers (India)")
    HELPLINES = [
        ("112", "National Emergency Number", "Police, fire and medical emergencies, 24x7", "🚨"),
        ("108", "Emergency Ambulance", "Free ambulance service available in most states", "🚑"),
        ("102", "Ambulance (mother and child)", "Free patient transport, mainly for pregnant women and infants", "🤱"),
        ("104", "Health Information Helpline", "State-run medical advice line, availability varies by state", "☎️"),
        ("14416", "Tele-MANAS", "Free 24x7 tele mental health counselling", "🧠"),
        ("14555", "Ayushman Bharat (PM-JAY)", "Help with the government health insurance scheme", "🏥"),
    ]
    cols = st.columns(3)
    for i, (num, name, desc, icon) in enumerate(HELPLINES):
        with cols[i % 3]:
            st.markdown(
                f'<div class="helpline"><div>{icon} <span class="name">{name}</span></div>'
                f'<div class="num"><a href="tel:{num}">📞 {num}</a></div>'
                f'<div class="desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption("Numbers can differ by state. Please verify locally. Tap a number on a phone to call.")

    section("Free online doctor consultation")
    t1, t2 = st.columns(2)
    with t1:
        st.link_button("💻 eSanjeevani (Govt. of India telemedicine)", "https://esanjeevani.mohfw.gov.in", use_container_width=True)
    with t2:
        st.link_button("🏥 Ayushman Bharat PM-JAY", "https://pmjay.gov.in", use_container_width=True)

    section("Which doctor should I see?")
    SPECIALISTS = {
        "Diabetes or high blood sugar": {
            "doctor": "Diabetologist or Endocrinologist (a General Physician is a good first step)",
            "search": "diabetologist",
            "tests": ["Fasting and post-meal blood glucose", "HbA1c", "Lipid profile", "Kidney function tests"],
            "carry": ["Previous blood reports", "List of current medicines", "Home glucose readings, if any"],
        },
        "Heart disease or chest pain": {
            "doctor": "Cardiologist (go to the emergency room for severe or sudden chest pain)",
            "search": "cardiologist",
            "tests": ["ECG", "2D Echo", "Lipid profile", "Treadmill (stress) test"],
            "carry": ["Previous ECG and reports", "List of current medicines", "Notes on when symptoms occur"],
        },
        "Parkinson's, tremors or movement problems": {
            "doctor": "Neurologist (a movement disorder specialist if available)",
            "search": "neurologist",
            "tests": ["Neurological examination", "MRI brain, if advised", "Blood tests to rule out other causes"],
            "carry": ["Short video of the tremor or walking", "List of current medicines", "Family history details"],
        },
        "General check-up": {
            "doctor": "General Physician or Family Doctor",
            "search": "general physician",
            "tests": ["Complete blood count", "Blood sugar", "Lipid profile", "Blood pressure check"],
            "carry": ["Previous reports", "List of current medicines", "Your questions written down"],
        },
    }
    c1, c2 = st.columns([2, 1])
    with c1:
        concern = st.selectbox("What is your concern?", list(SPECIALISTS.keys()))
    with c2:
        city = st.text_input("Your city (optional)", placeholder="e.g. Bhopal")
    info = SPECIALISTS[concern]

    left, right = st.columns(2)
    with left:
        card(
            "👨‍⚕️",
            "Recommended specialist",
            [info["doctor"], "<b>Tests the doctor may suggest:</b> " + ", ".join(info["tests"])],
        )
    with right:
        card("🎒", "What to carry", ["• " + item for item in info["carry"]])

    where = f"{info['search']} in {city.strip()}" if city.strip() else f"{info['search']} near me"
    st.link_button(
        f"🗺️ Find a {info['search']} on Google Maps",
        f"https://www.google.com/maps/search/?api=1&query={quote_plus(where)}",
        use_container_width=True,
    )

    section("Warning signs: seek emergency care immediately")
    with st.expander("❤️ Possible heart attack"):
        st.markdown(
            "- Chest pain, pressure or tightness lasting more than a few minutes\n"
            "- Pain spreading to the arm, jaw, neck or back\n"
            "- Sweating, nausea, breathlessness or sudden dizziness\n\n"
            "**Call 112 or 108 right away. Do not drive yourself.**"
        )
    with st.expander("🧠 Possible stroke (think FAST)"):
        st.markdown(
            "- **F**ace drooping on one side\n"
            "- **A**rm weakness or numbness\n"
            "- **S**peech slurred or confused\n"
            "- **T**ime to call 112 or 108 immediately"
        )
    with st.expander("🩸 Diabetic emergency"):
        st.markdown(
            "- **Very low sugar:** shaking, sweating, confusion, fainting. Give fast sugar if the person is awake and call for help.\n"
            "- **Very high sugar:** extreme thirst, frequent urination, fruity breath, vomiting, drowsiness. Seek emergency care."
        )

    section("Doctor visit checklist")
    checklist = [
        "Write down your symptoms and when they started",
        "Carry previous reports and prescriptions",
        "Note any medicines or supplements you take",
        "Prepare your questions for the doctor",
        "Download your report from the My Reports page",
    ]
    done = sum(st.checkbox(item, key=f"chk_{i}") for i, item in enumerate(checklist))
    st.progress(done / len(checklist), text=f"{done} of {len(checklist)} ready")
    disclaimer()


# ===========================================================================
# HEALTH LIBRARY
# ===========================================================================
if selected == "Health Library":
    hero(
        "📚 Health Library",
        "Simple, reliable guidance on prevention, warning signs and common questions.",
    )

    tab_d, tab_h, tab_p, tab_faq = st.tabs(["🩸 Diabetes", "❤️ Heart", "🧠 Parkinson's", "❓ FAQ"])

    with tab_d:
        c1, c2 = st.columns(2)
        with c1:
            card(
                "🥗",
                "Prevention and lifestyle",
                [
                    "• Keep a healthy weight, even a 5 to 7% loss helps",
                    "• Choose whole grains, vegetables, pulses and fibre",
                    "• Reduce sugary drinks and refined carbohydrates",
                    "• Stay active for 150 minutes a week",
                    "• Get fasting sugar or HbA1c checked yearly if at risk",
                ],
            )
        with c2:
            card(
                "⚠️",
                "Common warning signs",
                [
                    "• Increased thirst and frequent urination",
                    "• Unexplained weight loss and constant tiredness",
                    "• Blurred vision and slow-healing wounds",
                    "• Tingling or numbness in hands and feet",
                ],
            )

    with tab_h:
        c1, c2 = st.columns(2)
        with c1:
            card(
                "🏃",
                "Prevention and lifestyle",
                [
                    "• Do not smoke and avoid tobacco",
                    "• Limit salt, fried food and trans fats",
                    "• Exercise regularly and manage stress",
                    "• Keep blood pressure and cholesterol in range",
                    "• Sleep 7 to 8 hours each night",
                ],
            )
        with c2:
            card(
                "⚠️",
                "Common warning signs",
                [
                    "• Chest discomfort, pressure or tightness",
                    "• Shortness of breath during light activity",
                    "• Unusual fatigue, palpitations or dizziness",
                    "• Swelling in the feet or ankles",
                ],
            )

    with tab_p:
        c1, c2 = st.columns(2)
        with c1:
            card(
                "🧘",
                "Living well and staying active",
                [
                    "• Regular exercise helps balance and movement",
                    "• Physiotherapy and speech therapy can help",
                    "• Eat a balanced, fibre-rich diet",
                    "• Treatment plans should come from a neurologist",
                ],
            )
        with c2:
            card(
                "⚠️",
                "Common early signs",
                [
                    "• Tremor in a hand, finger or chin at rest",
                    "• Slowed movement and stiffness",
                    "• Smaller handwriting and softer voice",
                    "• Reduced facial expression or balance problems",
                ],
            )

    with tab_faq:
        with st.expander("Is this a medical diagnosis?"):
            st.write("No. This is an educational screening tool. Only a qualified doctor can diagnose a condition after examination and tests.")
        with st.expander("How accurate is the prediction?"):
            st.write("The models were trained on public datasets and perform reasonably on them, but real-world accuracy can differ. Treat the output as a prompt to talk to a doctor, not as a final answer.")
        with st.expander("Is my data stored?"):
            st.write("The values you enter are used only to run the prediction in your current browser session. They are not saved to a database. Your reports can be downloaded to your own device.")
        with st.expander("Why does the app use these particular inputs?"):
            st.write("Each model uses the same clinical features it was trained on, such as glucose and BMI for diabetes, ECG and cholesterol for heart disease, and voice measures for Parkinson's.")
        with st.expander("What should I do if the result says higher likelihood?"):
            st.write("Do not panic. Book an appointment with the suitable specialist (see Doctor Helpline) and carry your downloaded report. If you have severe symptoms, seek emergency care.")
    disclaimer()


# ===========================================================================
# MY REPORTS
# ===========================================================================
if selected == "My Reports":
    hero(
        "📄 My Reports",
        "All screenings from this session. Download a report to share with your doctor.",
    )

    history = st.session_state.history
    if not history:
        st.info("No screenings yet. Run a prediction and it will appear here.")
    else:
        df = pd.DataFrame(
            [
                {"Time": r["time"], "Patient": r["patient"], "Test": r["test"], "Result": r["result"]}
                for r in history
            ]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download summary (CSV)",
                df.to_csv(index=False),
                file_name="screening_summary.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.session_state.results = {}
                st.rerun()

        section("Download a full report")
        labels = [f"{i + 1}. {r['test']} · {r['time']}" for i, r in enumerate(history)]
        choice = st.selectbox("Choose a screening", labels)
        rec = history[labels.index(choice)]
        st.text(build_report(rec))
        st.download_button(
            "⬇️ Download this report",
            build_report(rec),
            file_name=f"{rec['test'].lower().replace(' ', '_')}_report.txt",
            mime="text/plain",
            key="dl_selected",
        )
    disclaimer()


# ===========================================================================
# ABOUT
# ===========================================================================
if selected == "About":
    hero(
        "ℹ️ About this Project",
        "A machine learning web app for early health screening, built as a learning project.",
    )

    c1, c2 = st.columns(2)
    with c1:
        card(
            "🧪",
            "Models and data",
            [
                "<b>Diabetes:</b> SVM (linear kernel), Pima Indians Diabetes dataset",
                "<b>Heart disease:</b> Logistic Regression, UCI Heart Disease dataset",
                "<b>Parkinson's:</b> SVM (linear kernel), UCI Parkinson's voice dataset",
            ],
        )
    with c2:
        card(
            "🛠️",
            "Tech stack",
            [
                "Python, Pandas, NumPy",
                "scikit-learn for model training",
                "Pickle for model storage",
                "Streamlit for the web interface",
            ],
        )

    section("Limitations")
    st.markdown(
        "- Models are trained on small public datasets and may not generalise to every population.\n"
        "- Results are probabilistic indicators, not diagnoses.\n"
        "- Reference ranges shown are general adult values.\n"
        "- This app does not replace a doctor's examination."
    )
    if DEVELOPER_NAME:
        st.markdown(f"**Developer:** {DEVELOPER_NAME}")
    if GITHUB_URL:
        st.link_button("View source code on GitHub", GITHUB_URL)
    disclaimer()
