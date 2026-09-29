import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/insurance_gradient_boosting_model.pkl"
    )

    feature_columns = joblib.load(
        "models/insurance_feature_columns.pkl"
    )

    return model, feature_columns


model, feature_columns = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ==============================
       GLOBAL
       ============================== */

    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(99, 102, 241, 0.15),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 10%,
                rgba(14, 165, 233, 0.10),
                transparent 25%
            ),
            #070a12;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ==============================
       REMOVE STREAMLIT CHROME
       ============================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ==============================
       HERO
       ============================== */

    .hero {
        padding: 38px 42px;
        border-radius: 24px;

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.92),
                rgba(15, 23, 42, 0.95)
            );

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 20px 60px rgba(0,0,0,0.30);

        margin-bottom: 28px;
    }

    .badge {
        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(99,102,241,0.13);

        border:
            1px solid rgba(129,140,248,0.25);

        color: #a5b4fc;

        font-size: 12px;

        font-weight: 700;

        letter-spacing: 0.7px;

        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 42px;

        font-weight: 800;

        letter-spacing: -1.5px;

        color: #f8fafc;

        margin-bottom: 10px;
    }

    .hero-subtitle {
        color: #94a3b8;

        font-size: 16px;

        line-height: 1.6;

        max-width: 720px;
    }


    /* ==============================
       SECTION TITLES
       ============================== */

    .section-title {
        color: #f1f5f9;

        font-size: 20px;

        font-weight: 750;

        margin-top: 28px;

        margin-bottom: 14px;
    }


    /* ==============================
       PERFORMANCE CARDS
       ============================== */

    .metric-card {
        padding: 20px;

        min-height: 105px;

        border-radius: 18px;

        background:
            rgba(15,23,42,0.78);

        border:
            1px solid rgba(255,255,255,0.07);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.20);
    }

    .metric-label {
        color: #64748b;

        font-size: 12px;

        margin-bottom: 8px;
    }

    .metric-value {
        color: #f8fafc;

        font-size: 21px;

        font-weight: 750;
    }


    /* ==============================
       INPUT SECTION
       ============================== */

    .input-card {
        padding: 24px;

        border-radius: 20px;

        background:
            rgba(15,23,42,0.72);

        border:
            1px solid rgba(255,255,255,0.07);

        box-shadow:
            0 15px 40px rgba(0,0,0,0.22);
    }


    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {

        width: 100%;

        min-height: 52px;

        border-radius: 12px;

        border: 0;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #4f46e5
            );

        color: white;

        font-size: 16px;

        font-weight: 700;

        box-shadow:
            0 8px 25px rgba(79,70,229,0.30);

        transition: 0.2s ease;
    }

    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 12px 32px rgba(79,70,229,0.45);
    }


    /* ==============================
       PREDICTION
       ============================== */

    .prediction-card {

        margin-top: 25px;

        padding: 34px;

        text-align: center;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(16,185,129,0.12),
                rgba(6,78,59,0.20)
            );

        border:
            1px solid rgba(52,211,153,0.30);

        box-shadow:
            0 15px 45px rgba(0,0,0,0.25);
    }

    .prediction-label {

        color: #6ee7b7;

        font-size: 13px;

        font-weight: 700;

        letter-spacing: 1px;

        margin-bottom: 10px;
    }

    .prediction-value {

        color: #ecfdf5;

        font-size: 44px;

        font-weight: 850;

        letter-spacing: -1px;
    }

    .prediction-note {

        color: #94a3b8;

        font-size: 13px;

        margin-top: 10px;
    }


    /* ==============================
       FOOTER
       ============================== */

    .custom-footer {

        text-align: center;

        color: #64748b;

        font-size: 12px;

        margin-top: 45px;

        padding-top: 20px;

        border-top:
            1px solid rgba(255,255,255,0.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="badge">
            MACHINE LEARNING REGRESSION SYSTEM
        </div>

        <div class="hero-title">
            Insurance Charges Predictor
        </div>

        <div class="hero-subtitle">
            Estimate individual medical insurance charges
            using a tuned Gradient Boosting regression model.
        </div>

    </div>
    """
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-label">Algorithm</div>
            <div class="metric-value">Gradient Boosting</div>
        </div>
        """
    )


with c2:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">90.25%</div>
        </div>
        """
    )


with c3:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">₹2,449.85</div>
        </div>
        """
    )


with c4:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">₹4,233.48</div>
        </div>
        """
    )


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Customer Information</div>',
    unsafe_allow_html=True
)

st.html(
    """
    <div class="input-card"></div>
    """
)

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )


with col2:

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )


# ============================================================
# PREDICT
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Insurance Charges"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })


    input_data = pd.get_dummies(
        input_data,
        columns=[
            "sex",
            "smoker",
            "region"
        ],
        drop_first=True
    )


    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    prediction = model.predict(input_data)[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.html(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                ESTIMATED INSURANCE CHARGES
            </div>

            <div class="prediction-value">
                ₹{prediction:,.2f}
            </div>

            <div class="prediction-note">
                Estimated using the trained Gradient Boosting model
            </div>

        </div>
        """
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Summary</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("Age", age)

    with s2:
        st.metric("BMI", f"{bmi:.1f}")

    with s3:
        st.metric("Smoker", smoker.title())

    with s4:
        st.metric("Children", children)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="custom-footer">

        Insurance Charges Prediction System
        <br><br>

        Machine Learning Project
        &nbsp;•&nbsp;
        Gradient Boosting Regression

    </div>
    """
)