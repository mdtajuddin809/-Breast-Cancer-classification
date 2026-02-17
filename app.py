import streamlit as st
import numpy as np
import joblib
# from tensorflow.keras.models import load_model  # Uncomment when deploying

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CancerScan · AI Diagnostic",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  /* ── Reset & Base ── */
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .stApp {
    background: #0a0d12;
    color: #e8eaf0;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* ── Hero Banner ── */
  .hero {
    background: linear-gradient(135deg, #0a0d12 0%, #0d1520 40%, #091018 100%);
    border-bottom: 1px solid rgba(0,200,160,0.15);
    padding: 52px 60px 36px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,200,160,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.2em;
    color: #00c8a0;
    text-transform: uppercase;
    margin-bottom: 14px;
  }
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 5vw, 56px);
    font-weight: 400;
    color: #f0f4f8;
    line-height: 1.05;
    margin: 0 0 16px;
  }
  .hero-title em { font-style: italic; color: #00c8a0; }
  .hero-subtitle {
    font-size: 15px;
    font-weight: 300;
    color: #8a92a0;
    max-width: 480px;
    line-height: 1.65;
  }
  .hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,200,160,0.1);
    border: 1px solid rgba(0,200,160,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #00c8a0;
    margin-top: 20px;
  }

  /* ── Layout ── */
  .main-grid {
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 0;
    min-height: calc(100vh - 200px);
  }

  /* ── Sidebar Panel ── */
  .sidebar-panel {
    background: #0d1117;
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 32px 24px;
  }
  .sidebar-section {
    margin-bottom: 28px;
  }
  .sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    color: #4a5568;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0;
  }
  .info-key { font-size: 12px; color: #6b7280; }
  .info-val {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #00c8a0;
  }

  /* ── Feature Groups ── */
  .features-area { padding: 32px 40px; }
  .group-header {
    display: flex; align-items: center; gap: 12px;
    margin: 0 0 20px;
  }
  .group-number {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    width: 24px;
  }
  .group-title {
    font-size: 13px;
    font-weight: 500;
    color: #c8d0dc;
    letter-spacing: 0.04em;
  }
  .group-divider {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
  }

  /* ── Number Input Overrides ── */
  .stNumberInput > div > div > input {
    background: #111520 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    transition: border-color 0.2s !important;
  }
  .stNumberInput > div > div > input:focus {
    border-color: rgba(0,200,160,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,200,160,0.08) !important;
  }
  .stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: #6b7280 !important;
    font-weight: 400 !important;
  }

  /* ── Predict Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #00c8a0 0%, #00a882 100%) !important;
    color: #0a0d12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 14px 36px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,200,160,0.25) !important;
  }

  /* ── Result Cards ── */
  .result-malignant {
    background: linear-gradient(135deg, rgba(220,38,38,0.12) 0%, rgba(153,27,27,0.08) 100%);
    border: 1px solid rgba(220,38,38,0.3);
    border-left: 4px solid #dc2626;
    border-radius: 10px;
    padding: 24px 28px;
    margin-top: 20px;
  }
  .result-benign {
    background: linear-gradient(135deg, rgba(0,200,160,0.12) 0%, rgba(0,168,130,0.08) 100%);
    border: 1px solid rgba(0,200,160,0.3);
    border-left: 4px solid #00c8a0;
    border-radius: 10px;
    padding: 24px 28px;
    margin-top: 20px;
  }
  .result-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .result-label.bad { color: #f87171; }
  .result-label.good { color: #00c8a0; }
  .result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    font-weight: 400;
    line-height: 1.1;
    margin-bottom: 10px;
  }
  .result-title.bad { color: #fca5a5; }
  .result-title.good { color: #a7f3e0; }
  .result-desc { font-size: 13px; color: #8a92a0; line-height: 1.6; }
  .confidence-bar-wrap { margin-top: 16px; }
  .confidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    margin-bottom: 6px;
    display: flex; justify-content: space-between;
  }
  .confidence-track {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
  }
  .confidence-fill-bad  { height: 100%; background: #dc2626; border-radius: 2px; }
  .confidence-fill-good { height: 100%; background: #00c8a0; border-radius: 2px; }

  /* ── Disclaimer ── */
  .disclaimer {
    font-size: 11px;
    color: #3d4451;
    line-height: 1.6;
    padding: 16px 20px;
    background: rgba(255,255,255,0.02);
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.04);
    margin-top: 16px;
  }

  /* ── Section Chip ── */
  .chip {
    display: inline-block;
    background: rgba(0,200,160,0.08);
    border: 1px solid rgba(0,200,160,0.18);
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #00c8a0;
    margin-bottom: 20px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🔬 &nbsp; Neural Network Diagnostic System</div>
  <h1 class="hero-title">Breast Cancer<br><em>Risk Analysis</em></h1>
  <p class="hero-subtitle">
    Enter the 30 clinical measurements from a fine needle aspirate biopsy.
    The ANN model will classify the sample as benign or malignant.
  </p>
  <div class="hero-badge">
    <span>●</span> Model · Wisconsin Diagnostic Breast Cancer · 30 Features
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Feature Definitions ───────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "Mean Values": {
        "desc": "Average of each characteristic across all cells in the sample",
        "features": [
            ("radius_mean",          "Radius (mean)",           0.0, 30.0,  14.1),
            ("texture_mean",         "Texture (mean)",          0.0, 40.0,  19.3),
            ("perimeter_mean",       "Perimeter (mean)",        0.0, 200.0, 92.0),
            ("area_mean",            "Area (mean)",             0.0, 2500.0,654.9),
            ("smoothness_mean",      "Smoothness (mean)",       0.0, 0.2,   0.096),
            ("compactness_mean",     "Compactness (mean)",      0.0, 0.5,   0.104),
            ("concavity_mean",       "Concavity (mean)",        0.0, 0.5,   0.089),
            ("concave_points_mean",  "Concave Points (mean)",   0.0, 0.2,   0.049),
            ("symmetry_mean",        "Symmetry (mean)",         0.0, 0.4,   0.181),
            ("fractal_dim_mean",     "Fractal Dimension (mean)",0.0, 0.1,   0.063),
        ],
    },
    "SE Values": {
        "desc": "Standard error — variability of each characteristic",
        "features": [
            ("radius_se",            "Radius (SE)",             0.0, 3.0,   0.405),
            ("texture_se",           "Texture (SE)",            0.0, 4.0,   1.217),
            ("perimeter_se",         "Perimeter (SE)",          0.0, 22.0,  2.866),
            ("area_se",              "Area (SE)",               0.0, 550.0, 40.34),
            ("smoothness_se",        "Smoothness (SE)",         0.0, 0.04,  0.007),
            ("compactness_se",       "Compactness (SE)",        0.0, 0.15,  0.025),
            ("concavity_se",         "Concavity (SE)",          0.0, 0.4,   0.032),
            ("concave_points_se",    "Concave Points (SE)",     0.0, 0.06,  0.012),
            ("symmetry_se",          "Symmetry (SE)",           0.0, 0.08,  0.020),
            ("fractal_dim_se",       "Fractal Dimension (SE)",  0.0, 0.03,  0.004),
        ],
    },
    "Worst Values": {
        "desc": "Largest (worst) value of each characteristic in the sample",
        "features": [
            ("radius_worst",         "Radius (worst)",          0.0, 40.0,  16.27),
            ("texture_worst",        "Texture (worst)",         0.0, 50.0,  25.68),
            ("perimeter_worst",      "Perimeter (worst)",       0.0, 260.0,107.26),
            ("area_worst",           "Area (worst)",            0.0,4300.0, 880.6),
            ("smoothness_worst",     "Smoothness (worst)",      0.0, 0.25,  0.132),
            ("compactness_worst",    "Compactness (worst)",     0.0, 1.1,   0.254),
            ("concavity_worst",      "Concavity (worst)",       0.0, 1.3,   0.272),
            ("concave_points_worst", "Concave Points (worst)",  0.0, 0.3,   0.115),
            ("symmetry_worst",       "Symmetry (worst)",        0.0, 0.7,   0.290),
            ("fractal_dim_worst",    "Fractal Dimension (worst)",0.0,0.25,  0.084),
        ],
    },
}

# ─── Collect Features ──────────────────────────────────────────────────────────
col_form, col_result = st.columns([3, 1], gap="large")

with col_form:
    st.markdown('<div style="padding: 32px 40px 0;">', unsafe_allow_html=True)

    all_features = []
    for g_idx, (group_name, group_data) in enumerate(FEATURE_GROUPS.items(), 1):
        st.markdown(f"""
        <div class="group-header">
          <span class="group-number">0{g_idx}</span>
          <span class="group-title">{group_name.upper()}</span>
          <span style="font-size:12px;color:#4a5568;margin-left:8px;">— {group_data['desc']}</span>
          <div class="group-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(5)
        for i, (key, label, min_v, max_v, default) in enumerate(group_data["features"]):
            with cols[i % 5]:
                val = st.number_input(
                    label,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(default),
                    step=float((max_v - min_v) / 1000),
                    format="%.4f",
                    key=key,
                )
                all_features.append(val)

        st.markdown("<div style='margin-bottom:32px;'></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Result Panel ──────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div style="padding: 32px 24px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="chip">DIAGNOSIS OUTPUT</div>', unsafe_allow_html=True)

    predict_btn = st.button("Run Analysis →", use_container_width=True)

    if predict_btn:
        input_data = np.array([all_features])

        # ── Load model & scaler (comment out for demo) ──
        # model  = load_model("breast_cancer_ann.h5")
        # scaler = joblib.load("scaler.pkl")
        # input_scaled = scaler.transform(input_data)
        # prediction = model.predict(input_scaled)
        # result = float(prediction[0][0])

        # Demo mode: placeholder score
        result = float(np.clip(np.mean(input_data) / 100, 0, 1))

        pct = int(result * 100)

        if result > 0.5:
            st.markdown(f"""
            <div class="result-malignant">
              <div class="result-label bad">⚠ Prediction Result</div>
              <div class="result-title bad">Malignant</div>
              <p class="result-desc">
                The model indicates high likelihood of malignancy.
                Please consult an oncologist immediately.
              </p>
              <div class="confidence-bar-wrap">
                <div class="confidence-label">
                  <span>Confidence score</span>
                  <span>{pct}%</span>
                </div>
                <div class="confidence-track">
                  <div class="confidence-fill-bad" style="width:{pct}%;"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-benign">
              <div class="result-label good">✓ Prediction Result</div>
              <div class="result-title good">Benign</div>
              <p class="result-desc">
                The model indicates the sample is likely benign.
                Routine follow-up with a physician is still advised.
              </p>
              <div class="confidence-bar-wrap">
                <div class="confidence-label">
                  <span>Confidence score</span>
                  <span>{100 - pct}%</span>
                </div>
                <div class="confidence-track">
                  <div class="confidence-fill-good" style="width:{100 - pct}%;"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <strong style="color:#6b7280;">⚕ Clinical Disclaimer</strong><br>
      This tool is for research and educational purposes only.
      It does not constitute medical advice and must not replace
      professional clinical assessment. Always consult a licensed
      medical professional for diagnosis and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

    # ── Model info cards ──
    st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
    st.markdown('<div class="chip" style="margin-top:20px;">MODEL INFO</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:8px;">
      <div class="info-row">
        <span class="info-key">Architecture</span>
        <span class="info-val">ANN</span>
      </div>
      <div class="info-row">
        <span class="info-key">Input features</span>
        <span class="info-val">30</span>
      </div>
      <div class="info-row">
        <span class="info-key">Output</span>
        <span class="info-val">Binary</span>
      </div>
      <div class="info-row">
        <span class="info-key">Dataset</span>
        <span class="info-val">WDBC</span>
      </div>
      <div class="info-row">
        <span class="info-key">Scaler</span>
        <span class="info-val">StandardScaler</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)