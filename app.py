# app.py
# Streamlit UI for CMAPSS RUL Prediction (FD001)
# Run: streamlit run app.py

import io
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- CONSULTING DARK HERO (FIXED) ---
st.set_page_config(page_title="Turbofan RUL Predictor", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
        text-align: center;
        padding: 3rem 1rem 2.5rem 1rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .hero p {
        font-size: 1.1rem;
        color: #d0d0d0;
        margin-top: 0.3rem;
    }
    .hero img {
        margin-top: 1.5rem;
        border-radius: 10px;
        width: 60%;
        max-width: 700px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
    }
    .metric {
        text-align: center;
    }
    .metric h3 {
        font-size: 1rem;
        color: #aaa;
        margin-bottom: 0.4rem;
    }
    .metric p {
        font-size: 1.4rem;
        font-weight: 600;
        color: #fff;
    }
    </style>

    <div class="hero">
        <h1>✈️ Turbofan Remaining Useful Life Predictor</h1>
        <p>Predictive Maintenance · NASA CMAPSS · Gradient Boosting Regressor</p>
        <img src="https://raw.githubusercontent.com/isawgod/turbofan-rul-prediction/main/assets/app_screenshot.png" alt="App Screenshot">

        <div class="metric-container">
            <div class="metric">
                <h3>Model</h3>
                <p>Gradient Boosting</p>
            </div>
            <div class="metric">
                <h3>Mean Abs. Error</h3>
                <p>≈ 6.38 cycles</p>
            </div>
            <div class="metric">
                <h3>Dataset</h3>
                <p>NASA CMAPSS FD001</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Local imports (support both module and script run) ---
try:
    # If running as a module (recommended with "python -m"), this works:
    from src.features import FEATURES as MODEL_FEATURES, make_features_from_raw
except Exception:
    # Fallback for "streamlit run app.py" executed from project root
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    from src.features import FEATURES as MODEL_FEATURES, make_features_from_raw


st.set_page_config(page_title="Turbofan RUL Predictor", page_icon="✈️", layout="wide")

# --- Paths / Model ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "rul_gb_fd001.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()


# --- Helpers ---
def looks_like_raw(df: pd.DataFrame) -> bool:
    sensors = [f"s{i:02d}" for i in range(1, 22)]
    return ("unit" in df.columns and "cycle" in df.columns and all(c in df.columns for c in sensors))

def looks_like_features(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in MODEL_FEATURES)

def ensure_features(df: pd.DataFrame, auto_fe: bool) -> pd.DataFrame:
    """
    If df is raw and auto_fe is True -> compute features.
    If df already has features -> return selection.
    """
    if looks_like_features(df):
        return df[MODEL_FEATURES].copy()
    if looks_like_raw(df) and auto_fe:
        return make_features_from_raw(df)
    if looks_like_raw(df) and not auto_fe:
        st.warning("Detected RAW CMAPSS schema but Auto-Feature is OFF. Enable the toggle below.")
        st.stop()
    st.error("Input does not match RAW schema (unit, cycle, s01..s21) nor Feature schema.")
    st.stop()


# --- UI ---
st.title("✈️ Turbofan Engine RUL Predictor (NASA CMAPSS – FD001)")
st.markdown(
    "Upload a CSV with either **RAW CMAPSS rows** (`unit, cycle, s01..s21`) "
    "or **pre-computed features**. The app will predict Remaining Useful Life (RUL)."
)

with st.sidebar:
    st.header("Settings")
    auto_fe = st.toggle("Auto feature engineering (for RAW input)", value=True,
                        help="Turn on to compute features from raw CMAPSS rows.")
    st.caption("Model file expected at: `models/rul_gb_fd001.joblib`")
    st.markdown("**Expected feature columns:**")
    st.code(", ".join(MODEL_FEATURES), language="bash")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(data.head(20), use_container_width=True)

    # Build features
    feats = ensure_features(data, auto_fe=auto_fe)

    # Predict
    preds = model.predict(feats)
    out = data.copy()
    out["RUL_pred"] = preds

    st.subheader("Predictions (head)")
    st.dataframe(out.head(20), use_container_width=True)

    # Download button
    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_buf.getvalue(),
        file_name="predictions.csv",
        mime="text/csv",
    )

    # If ground-truth RUL present -> quick evaluation and plots
    if "RUL" in out.columns:
        st.markdown("---")
        st.subheader("Evaluation (Ground Truth Provided)")
        y_true = out["RUL"].values
        y_pred = out["RUL_pred"].values
        mae = (abs(y_true - y_pred)).mean()
        st.metric("MAE (cycles)", f"{mae:,.2f}")

        # Plot: True vs Pred (index)
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.scatter(range(len(y_true)), y_true, label="True RUL", alpha=0.6)
        ax1.scatter(range(len(y_pred)), y_pred, label="Predicted RUL", alpha=0.6)
        ax1.set_title("True vs Predicted RUL")
        ax1.set_xlabel("Sample index")
        ax1.set_ylabel("RUL (cycles)")
        ax1.legend()
        st.pyplot(fig1)

        # Plot: Error histogram
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        errors = y_true - y_pred
        ax2.hist(errors, bins=30)
        ax2.set_title("Prediction Error Distribution (RUL)")
        ax2.set_xlabel("Error (cycles)")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

else:
    st.info("Upload a CSV to begin. RAW: `unit, cycle, s01..s21` • Features: see sidebar.")


# Footer tips
with st.expander("ℹ️ Tips & expected columns"):
    st.write("""
**RAW schema (minimum):**
- `unit`, `cycle`, `s01` ... `s21` (settings are optional).

**Feature schema (exact):**
- {}
    """.format(", ".join(MODEL_FEATURES)))
