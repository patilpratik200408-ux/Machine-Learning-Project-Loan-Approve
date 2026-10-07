import streamlit as st
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)


# =========================================================
# LOAD MODEL & SCALER
# =========================================================

@st.cache_resource
def load_files():

    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_files()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(14, 165, 233, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 85%,
                rgba(6, 182, 212, 0.14),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #071525 0%,
                #0d2942 52%,
                #071a2d 100%
            );

        background-attachment: fixed;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    /* MAIN TITLE COLOR */
    h1 {
        color: #67e8f9 !important;
        text-align: center;
        font-weight: 800 !important;
    }

    h2,
    h3 {
        color: #ffffff !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.97);
        border: 1px solid rgba(255, 255, 255, 0.75);
        border-radius: 18px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stVerticalBlockBorderWrapper"] h2,
    [data-testid="stVerticalBlockBorderWrapper"] h3,
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] label {
        color: #0f172a !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
    }

    [data-baseweb="input"],
    [data-baseweb="select"] {
        border-radius: 10px;
    }

    .stButton > button {
        height: 52px;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(
            90deg,
            #0284c7,
            #06b6d4
        );
        color: white;
        font-size: 16px;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.30);
    }

    .stButton > button:hover {
        border: 0;
        color: white;
    }

    [data-testid="stAlert"] {
        border-radius: 16px;
    }

    .footer-text {
        color: #cbd5e1 !important;
        text-align: center;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO SECTION
# =========================================================

st.title("🏦 Loan Approval Predictor")

st.markdown("### Finance • Machine Learning")

st.markdown(
    "Predict loan approval status using applicant financial "
    "and credit information."
)

st.caption(
    "🟢 Model Ready • Decision Tree Classification"
)


# =========================================================
# APPLICANT INFORMATION
# =========================================================

with st.container(border=True):

    st.subheader("👤 Applicant Information")

    st.caption(
        "Personal, education and employment details"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=0
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

    with col2:

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            value=10
        )

    with col3:

        cibil_score = st.slider(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700
        )

        st.caption(
            "Credit score range: 300 – 900"
        )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.write("")

with st.container(border=True):

    st.subheader("💰 Financial Information")

    st.caption(
        "Income, loan and asset information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=5000000,
            step=100000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=10000000,
            step=100000
        )

    with col2:

        residential_assets = st.number_input(
            "Residential Assets Value",
            min_value=0,
            value=5000000,
            step=100000
        )

        commercial_assets = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            value=2000000,
            step=100000
        )

    with col3:

        luxury_assets = st.number_input(
            "Luxury Assets Value",
            min_value=0,
            value=3000000,
            step=100000
        )

        bank_assets = st.number_input(
            "Bank Assets Value",
            min_value=0,
            value=2000000,
            step=100000
        )


# =========================================================
# ONE HOT ENCODING
# =========================================================

education_graduate = (
    1 if education == "Graduate" else 0
)

education_not_graduate = (
    1 if education == "Not Graduate" else 0
)

self_employed_no = (
    1 if self_employed == "No" else 0
)

self_employed_yes = (
    1 if self_employed == "Yes" else 0
)


# =========================================================
# INPUT DATAFRAME
# IMPORTANT:
# COLUMN ORDER MUST MATCH TRAINING DATA
# =========================================================

input_data = pd.DataFrame([{

    "no_of_dependents": dependents,

    "income_annum": income,

    "loan_amount": loan_amount,

    "loan_term": loan_term,

    "cibil_score": cibil_score,

    "residential_assets_value": residential_assets,

    "commercial_assets_value": commercial_assets,

    "luxury_assets_value": luxury_assets,

    "bank_asset_value": bank_assets,

    "education_ Graduate": education_graduate,

    "education_ Not Graduate": education_not_graduate,

    "self_employed_ No": self_employed_no,

    "self_employed_ Yes": self_employed_yes

}])


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

left, center, right = st.columns([1, 2, 1])

with center:

    predict_button = st.button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    prediction_text = str(prediction).strip().lower()


    # =====================================================
    # RESULT
    # =====================================================

    st.write("")

    st.subheader("📊 Prediction Result")

    if prediction_text == "approved":

        st.success(
            "### ✅ LOAN APPROVED\n\n"
            "The model predicts that this loan application "
            "is likely to be approved."
        )

    else:

        st.error(
            "### ❌ LOAN REJECTED\n\n"
            "The model predicts that this loan application "
            "is likely to be rejected."
        )


    # =====================================================
    # APPLICATION SUMMARY
    # =====================================================

    with st.container(border=True):

        st.subheader("📋 Application Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.caption("🎓 Education")
            st.write(education)

        with s2:

            st.caption("💼 Employment")

            if self_employed == "Yes":
                st.write("Self Employed")
            else:
                st.write("Salaried")

        with s3:

            st.caption("💳 CIBIL Score")
            st.write(cibil_score)

        with s4:

            st.caption("💰 Loan Amount")

            st.write(
                f"₹ {loan_amount:,.0f}"
            )


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.markdown(
    "<p class='footer-text'>"
    "🏦 Loan Approval Prediction • "
    "Machine Learning Project • "
    "Decision Tree Classifier"
    "</p>",
    unsafe_allow_html=True
)