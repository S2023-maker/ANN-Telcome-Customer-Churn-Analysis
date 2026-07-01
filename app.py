import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import onnxruntime as rt

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .block-container { padding: 2rem 3rem; }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e3a5f 100%);
        border: 1px solid #2a4a7f;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .header-box h1 { color: #e8f4fd; font-size: 2rem; font-weight: 700; margin: 0; }
    .header-box p  { color: #8ab4d4; font-size: 0.95rem; margin: 0.4rem 0 0; }

    /* Section titles */
    .section-title {
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.75rem;
    }

    /* Result cards */
    .result-churn {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-no-churn {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-label { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-sub   { font-size: 0.9rem; opacity: 0.8; }

    /* Metric boxes */
    .metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
    .metric-box {
        flex: 1;
        background: #1a1f2e;
        border: 1px solid #2a3a5a;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
    .metric-lbl { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

    /* Info banner */
    .info-banner {
        background: #1a1f2e;
        border-left: 3px solid #60a5fa;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 1rem;
    }

    /* Divider */
    hr { border-color: #1f2937; margin: 1.5rem 0; }

    /* Streamlit overrides */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    le           = joblib.load("artifacts/label_encoder.joblib")
    model        = rt.InferenceSession("artifacts/churn_model.onnx")
    with open("artifacts/metadata.json") as f:
        meta = json.load(f)
    return preprocessor, le, model, meta

try:
    preprocessor, le, model, meta = load_artifacts()
    threshold    = meta["threshold"]
    test_acc     = meta.get("test_accuracy", None)
    test_auc     = meta.get("test_auc", None)
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_err = str(e)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>📡 Telecom Churn Predictor</h1>
    <p>Enter customer details to predict whether they are likely to churn.
    Model: ANN · Binary classification · Threshold-tuned for recall.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_ok:
    st.error(f"⚠️ Could not load model artifacts. Make sure the `artifacts/` folder is in the same directory as `app.py`.\n\n`{artifact_err}`")
    st.stop()


# ─── Sidebar: model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    if test_acc:
        st.metric("Test Accuracy",  f"{test_acc:.2%}")
    if test_auc:
        st.metric("Test ROC-AUC",   f"{test_auc:.4f}")
    st.metric("Decision Threshold", f"{threshold:.2f}")
    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown("Dense 32 → 16 → 8 → 1 (sigmoid)")
    st.markdown("L2 regularization + Dropout")
    st.markdown("Trained with class-weighted loss")
    st.markdown("---")
    st.caption("v1.0 · Telco Churn Dataset")


# ─── Input form ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender          = st.selectbox("Gender",          ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner         = st.selectbox("Partner",         ["No", "Yes"])
    dependents      = st.selectbox("Dependents",      ["No", "Yes"])
    tenure_months   = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    phone_service   = st.selectbox("Phone Service",   ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
    internet_svc    = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])

with col3:
    device_protection = st.selectbox("Device Protection",  ["No", "Yes", "No internet service"])
    tech_support      = st.selectbox("Tech Support",       ["No", "Yes", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV",       ["No", "Yes", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies",   ["No", "Yes", "No internet service"])

st.markdown('<div class="section-title">Billing & Contract</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    contract          = st.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing",["Yes", "No"])

with col5:
    payment_method    = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col6:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",   min_value=0.0, max_value=10000.0, value=780.0, step=10.0)

st.markdown('<div class="section-title">Location</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    state = st.text_input("State", value="California")
with col8:
    city  = st.text_input("City",  value="Los Angeles")


# ─── Predict ─────────────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔍 Predict Churn")

if predict_btn:
    input_data = pd.DataFrame([{
        "state":               state,
        "city":                city,
        "gender":              gender,
        "senior_citizen":      senior_citizen,
        "partner":             partner,
        "dependents":          dependents,
        "tenure_months":       tenure_months,
        "phone_service":       phone_service,
        "multiple_lines":      multiple_lines,
        "internet_service":    internet_svc,
        "online_security":     online_security,
        "online_backup":       online_backup,
        "device_protection":   device_protection,
        "tech_support":        tech_support,
        "streaming_tv":        streaming_tv,
        "streaming_movies":    streaming_movies,
        "contract":            contract,
        "paperless_billing":   paperless_billing,
        "payment_method":      payment_method,
        "monthly_charges":     monthly_charges,
        "total_charges":       str(total_charges),
    }])

    try:
        X_processed = preprocessor.transform(input_data)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        X_processed = X_processed.astype(np.float32)
        input_name   = model.get_inputs()[0].name
        prob         = float(model.run(None, {input_name: X_processed})[0].ravel()[0])
        label        = "Yes" if prob >= threshold else "No"

        st.markdown("---")
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        rcol1, rcol2 = st.columns([1, 1])

        with rcol1:
            if label == "Yes":
                st.markdown(f"""
                <div class="result-churn">
                    <div class="result-label" style="color:#fca5a5;">⚠️ High Churn Risk</div>
                    <div class="result-sub" style="color:#fca5a5;">This customer is likely to churn</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-no-churn">
                    <div class="result-label" style="color:#86efac;">✅ Low Churn Risk</div>
                    <div class="result-sub" style="color:#86efac;">This customer is likely to stay</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{prob:.1%}</div>
                    <div class="metric-lbl">Churn Probability</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{threshold:.2f}</div>
                    <div class="metric-lbl">Threshold Used</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rcol2:
            st.markdown("**Probability Breakdown**")
            st.progress(prob)

            if prob >= 0.7:
                risk_level = "🔴 Very High — immediate retention action recommended"
            elif prob >= threshold:
                risk_level = "🟠 High — flag for outreach"
            elif prob >= 0.3:
                risk_level = "🟡 Moderate — monitor closely"
            else:
                risk_level = "🟢 Low — no action needed"

            st.markdown(f"**Risk level:** {risk_level}")

            key_factors = []
            if contract == "Month-to-month":
                key_factors.append("Month-to-month contract (highest churn rate)")
            if internet_svc == "Fiber optic":
                key_factors.append("Fiber optic internet (associated with higher churn)")
            if tenure_months <= 12:
                key_factors.append(f"Short tenure ({tenure_months} months)")
            if payment_method == "Electronic check":
                key_factors.append("Electronic check payment method")

            if key_factors:
                st.markdown("**Contributing factors detected:**")
                for f in key_factors:
                    st.markdown(f"• {f}")

        st.markdown(f"""
        <div class="info-banner">
            ℹ️ Predictions use a tuned threshold of {threshold:.2f} (instead of the default 0.5) to maximise recall on at-risk customers.
            Model test accuracy: {test_acc:.2%} · Test AUC: {test_auc:.4f}.
            A "High Churn Risk" flag means roughly 1 in 2 flagged customers actually churns (precision ~0.52),
            while catching ~75% of all true churners (recall ~0.75).
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure the column names and categories in the input match what the model was trained on.")
