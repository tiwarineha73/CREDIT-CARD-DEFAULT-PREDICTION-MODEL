"""
Credit Card Default Predictor — Production ML Web Application
Author: Neha Tiwari | GitHub: tiwarineha73
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime
import io

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:   #0a0e17;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --accent:       #3b82f6;
    --accent2:      #10b981;
    --danger:       #ef4444;
    --warning:      #f59e0b;
    --text-main:    #e2e8f0;
    --text-muted:   #64748b;
    --border:       #1e293b;
    --font:         'Sora', sans-serif;
    --mono:         'JetBrains Mono', monospace;
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-main) !important;
    font-family: var(--font) !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1424 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── Sidebar Nav ── */
.sidebar-logo {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    padding: 1rem 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
[data-testid="stSidebarNav"] li { padding: 0.2rem 0; }

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-success {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    padding: 1.5rem;
}
.card-danger {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px;
    padding: 1.5rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent);
    font-family: var(--mono);
    line-height: 1.1;
}
.metric-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.3rem;
}

/* ── Page heading ── */
.page-heading {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}
.page-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Form inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { 
    background-color: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
}

/* ── Risk Badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.risk-high   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.risk-medium { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.risk-low    { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }

/* ── Section tag ── */
.section-tag {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    color: var(--accent);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

/* ── Home hero ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -1px;
    color: #f8fafc;
}
.hero-title span { color: var(--accent); }
.hero-desc {
    font-size: 1rem;
    color: var(--text-muted);
    line-height: 1.7;
    margin: 1rem 0 1.5rem 0;
    max-width: 560px;
}

/* ── Stat strip on home ── */
.stat-strip {
    display: flex;
    gap: 1.5rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.stat-item { text-align: left; }
.stat-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    font-family: var(--mono);
}
.stat-desc {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    width: 100% !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 8px;
    gap: 0.25rem;
    padding: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Info/warning boxes ── */
.stAlert { border-radius: 8px !important; }

/* ── Plotly background patch ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS — Load assets
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.joblib"))

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE, "scaler.joblib"))

@st.cache_data
def load_metrics():
    with open(os.path.join(BASE, "metrics.json")) as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data.csv"))
    df = df.rename(columns={"default.payment.next.month": "DEFAULT"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
    return df

# Plotly theme shared config
PLOT_BG   = "#111827"
PLOT_GRID = "#1e293b"
PLOT_TEXT = "#94a3b8"
ACCENT    = "#3b82f6"
ACCENT2   = "#10b981"
DANGER    = "#ef4444"
WARNING   = "#f59e0b"

def chart_layout(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0", family="Sora"), x=0, xanchor="left"),
        height=height,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif", color=PLOT_TEXT),
        margin=dict(t=45, b=30, l=30, r=20),
        xaxis=dict(gridcolor=PLOT_GRID, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=PLOT_GRID, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">💳 CreditGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Credit Default Risk Platform</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🎯  Prediction", "📊  Data Analysis", "🧠  Model Insights", "📥  Report Download"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.72rem; color:#475569; line-height:1.8;'>
    <b style='color:#64748b;'>Dataset</b><br>
    UCI Credit Card Default<br>30,000 records · 23 features<br><br>
    <b style='color:#64748b;'>Model</b><br>
    Random Forest (200 trees)<br><br>
    <b style='color:#64748b;'>Built by</b><br>
    Neha Tiwari<br>
    <a href='https://github.com/tiwarineha73' style='color:#3b82f6;'>github.com/tiwarineha73</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("""
    <div class='hero-title'>
        Predict Credit Card<br><span>Default Risk</span><br>with Machine Learning
    </div>
    <div class='hero-desc'>
        CreditGuard AI uses a trained Random Forest model on 30,000 real-world 
        records to predict whether a customer will default on their next payment — 
        providing risk scores, explainability, and actionable insights.
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    metrics = load_metrics()
    c1, c2, c3, c4, c5 = st.columns(5)
    tiles = [
        (f"{metrics['accuracy']*100:.1f}%", "Accuracy"),
        (f"{metrics['precision']*100:.1f}%", "Precision"),
        (f"{metrics['recall']*100:.1f}%", "Recall"),
        (f"{metrics['f1']*100:.1f}%", "F1 Score"),
        (f"{metrics['auc']*100:.1f}%", "ROC-AUC"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], tiles):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='metric-val'>{val}</div>
            <div class='metric-lbl'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='section-tag'>How it works</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card card-accent'>
        <b style='color:#e2e8f0;'>🔢 Step 1 — Input</b><br>
        <span style='color:#94a3b8; font-size:0.85rem;'>Enter customer profile: credit limit, payment history, bill amounts, demographics.</span>
        </div>
        <div class='card card-accent'>
        <b style='color:#e2e8f0;'>⚙️ Step 2 — Predict</b><br>
        <span style='color:#94a3b8; font-size:0.85rem;'>The Random Forest model processes 23 features and outputs a default probability score.</span>
        </div>
        <div class='card card-accent'>
        <b style='color:#e2e8f0;'>📥 Step 3 — Report</b><br>
        <span style='color:#94a3b8; font-size:0.85rem;'>Download a full PDF-style text report with input data, risk level, and recommendations.</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-tag'>Dataset Overview</div>", unsafe_allow_html=True)
        df = load_data()
        total   = len(df)
        default = df["DEFAULT"].sum()
        rate    = default / total * 100

        st.markdown(f"""
        <div class='card'>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem;'>
            <div>
                <div style='font-size:1.5rem; font-weight:700; color:#3b82f6; font-family:JetBrains Mono;'>{total:,}</div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>Total Records</div>
            </div>
            <div>
                <div style='font-size:1.5rem; font-weight:700; color:#ef4444; font-family:JetBrains Mono;'>{default:,}</div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>Defaulted</div>
            </div>
            <div>
                <div style='font-size:1.5rem; font-weight:700; color:#f59e0b; font-family:JetBrains Mono;'>{rate:.1f}%</div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>Default Rate</div>
            </div>
            <div>
                <div style='font-size:1.5rem; font-weight:700; color:#10b981; font-family:JetBrains Mono;'>23</div>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>Features</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini donut
        fig = go.Figure(go.Pie(
            values=[total - default, default],
            labels=["No Default", "Default"],
            hole=0.7,
            marker_colors=[ACCENT2, DANGER],
            textinfo="none",
        ))
        fig.add_annotation(text=f"<b>{rate:.1f}%</b><br>default", x=0.5, y=0.5,
                           font=dict(size=14, color="#e2e8f0"), showarrow=False)
        chart_layout(fig, height=220)
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='background:rgba(59,130,246,0.06); border-color:rgba(59,130,246,0.2);'>
    <b style='color:#e2e8f0;'>⚠️ Disclaimer</b><br>
    <span style='color:#94a3b8; font-size:0.82rem;'>
    This is a portfolio project for educational and demonstration purposes.
    Predictions are based on historical patterns and should not be used for actual financial decisions.
    </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICTION
# ─────────────────────────────────────────────
elif page == "🎯  Prediction":
    st.markdown("<div class='page-heading'>🎯 Default Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Fill in the customer profile below and get an instant risk assessment.</div>", unsafe_allow_html=True)

    model  = load_model()

    with st.form("prediction_form"):
        # ── Section 1: Demographics ──
        st.markdown("<div class='section-tag'>Demographics & Profile</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35)
        with c2:
            sex = st.selectbox("Gender", ["Female (2)", "Male (1)"])
        with c3:
            education = st.selectbox("Education", [
                "Graduate (1)", "University (2)", "High School (3)", "Other (4)"
            ])
        with c4:
            marriage = st.selectbox("Marital Status", [
                "Married (1)", "Single (2)", "Other (3)"
            ])

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-tag'>Credit & Payment History</div>", unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            limit_bal = st.number_input("Credit Limit (NT dollar)", min_value=10000, max_value=1000000, value=50000, step=10000)
        with c6:
            pay_0 = st.selectbox("Repayment Status — Sep (PAY_0)",
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                index=2,
                help="-2=No consumption, -1=Paid fully, 0=Revolving, 1-8=Months delayed")

        c7, c8, c9, c10 = st.columns(4)
        with c7:
            pay_2 = st.selectbox("Pay Status Aug", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        with c8:
            pay_3 = st.selectbox("Pay Status Jul", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        with c9:
            pay_4 = st.selectbox("Pay Status Jun", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        with c10:
            pay_5 = st.selectbox("Pay Status May", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)

        pay_6 = st.selectbox("Repayment Status — Apr (PAY_6)", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-tag'>Bill Amounts (NT Dollar) — Last 6 Months</div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            bill1 = st.number_input("Bill Sep", value=10000, step=500)
            bill4 = st.number_input("Bill Jun", value=9000, step=500)
        with b2:
            bill2 = st.number_input("Bill Aug", value=9500, step=500)
            bill5 = st.number_input("Bill May", value=8500, step=500)
        with b3:
            bill3 = st.number_input("Bill Jul", value=9200, step=500)
            bill6 = st.number_input("Bill Apr", value=8000, step=500)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-tag'>Payment Amounts (NT Dollar) — Last 6 Months</div>", unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        with p1:
            pamt1 = st.number_input("Paid Sep", value=1000, step=100)
            pamt4 = st.number_input("Paid Jun", value=1000, step=100)
        with p2:
            pamt2 = st.number_input("Paid Aug", value=1000, step=100)
            pamt5 = st.number_input("Paid May", value=1000, step=100)
        with p3:
            pamt3 = st.number_input("Paid Jul", value=1000, step=100)
            pamt6 = st.number_input("Paid Apr", value=1000, step=100)

        submitted = st.form_submit_button("⚡ Run Prediction")

    # ── Process prediction ──
    if submitted:
        sex_val  = 1 if "Male" in sex else 2
        edu_val  = int(education.split("(")[1].replace(")", ""))
        mar_val  = int(marriage.split("(")[1].replace(")", ""))

        input_data = pd.DataFrame([{
            "LIMIT_BAL": limit_bal, "SEX": sex_val, "EDUCATION": edu_val,
            "MARRIAGE": mar_val, "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
            "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
            "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
            "PAY_AMT1": pamt1, "PAY_AMT2": pamt2, "PAY_AMT3": pamt3,
            "PAY_AMT4": pamt4, "PAY_AMT5": pamt5, "PAY_AMT6": pamt6,
        }])

        with st.spinner("Analyzing customer profile..."):
            prob     = model.predict_proba(input_data)[0][1]
            pred     = model.predict(input_data)[0]
            conf     = prob * 100 if pred == 1 else (1 - prob) * 100

        # Save to session for report
        st.session_state["last_prediction"] = {
            "inputs": input_data.to_dict(orient="records")[0],
            "prob": prob,
            "pred": pred,
            "conf": conf,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sex_label": sex, "edu_label": education,
            "mar_label": marriage,
        }

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Result card
        risk_pct = prob * 100
        if risk_pct >= 60:
            risk_label, risk_class, icon = "HIGH RISK", "risk-high", "🔴"
        elif risk_pct >= 35:
            risk_label, risk_class, icon = "MEDIUM RISK", "risk-medium", "🟡"
        else:
            risk_label, risk_class, icon = "LOW RISK", "risk-low", "🟢"

        result_class = "card-danger" if pred == 1 else "card-success"
        verdict      = "⚠️ Likely to Default" if pred == 1 else "✅ Unlikely to Default"
        verdict_clr  = "#ef4444" if pred == 1 else "#10b981"

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
            <div class='{result_class}'>
                <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>Prediction Result</div>
                <div style='font-size:1.8rem; font-weight:700; color:{verdict_clr}; margin-bottom:0.5rem;'>{verdict}</div>
                <span class='risk-badge {risk_class}'>{icon} {risk_label}</span>
                <div style='margin-top:1.2rem; display:grid; grid-template-columns:1fr 1fr; gap:1rem;'>
                    <div>
                        <div style='font-size:1.6rem; font-weight:700; color:{verdict_clr}; font-family:JetBrains Mono;'>{risk_pct:.1f}%</div>
                        <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>Default Probability</div>
                    </div>
                    <div>
                        <div style='font-size:1.6rem; font-weight:700; color:#e2e8f0; font-family:JetBrains Mono;'>{conf:.1f}%</div>
                        <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>Model Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0", "family": "Sora"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": PLOT_TEXT, "tickfont": {"size": 10}},
                    "bar":  {"color": DANGER if pred == 1 else ACCENT2, "thickness": 0.25},
                    "bgcolor": PLOT_BG,
                    "bordercolor": PLOT_GRID,
                    "steps": [
                        {"range": [0, 35],   "color": "rgba(16,185,129,0.15)"},
                        {"range": [35, 60],  "color": "rgba(245,158,11,0.15)"},
                        {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": DANGER, "width": 2}, "value": risk_pct},
                },
                domain={"x": [0, 1], "y": [0, 1]},
            ))
            fig.update_layout(height=230, paper_bgcolor=PLOT_BG, margin=dict(t=10, b=10, l=20, r=20),
                              font=dict(color=PLOT_TEXT))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Recommendations
        st.markdown("<div class='section-tag' style='margin-top:1rem;'>Recommendations</div>", unsafe_allow_html=True)
        if pred == 1:
            tips = [
                ("💳", "Review Credit Utilization", "Customer is likely over-leveraged. Consider reducing credit limit."),
                ("📅", "Monitor Payment History", "Late payments are a key driver. Implement payment reminders."),
                ("🔔", "Early Intervention", "Proactively contact the customer for restructuring options."),
                ("📊", "Periodic Re-assessment", "Re-evaluate every 30 days as risk can change quickly."),
            ]
        else:
            tips = [
                ("✅", "Good Standing", "Customer profile indicates responsible credit usage."),
                ("🎯", "Upsell Opportunity", "Consider offering higher credit limits or premium products."),
                ("📈", "Engagement", "Loyalty rewards may further reduce churn and default risk."),
                ("🔄", "Routine Monitoring", "Maintain standard periodic reviews every 90 days."),
            ]

        tip_cols = st.columns(4)
        for col, (icon, title, desc) in zip(tip_cols, tips):
            col.markdown(f"""
            <div class='card' style='height:140px;'>
                <div style='font-size:1.4rem;'>{icon}</div>
                <div style='font-weight:600; color:#e2e8f0; font-size:0.82rem; margin:0.3rem 0;'>{title}</div>
                <div style='color:#64748b; font-size:0.75rem; line-height:1.4;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='color:#475569; font-size:0.72rem; margin-top:0.5rem;'>
        ℹ️ Go to <b>Report Download</b> page to export a full PDF-style report of this prediction.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DATA ANALYSIS
# ─────────────────────────────────────────────
elif page == "📊  Data Analysis":
    st.markdown("<div class='page-heading'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Visual exploration of the UCI Credit Card Default dataset (30,000 records).</div>", unsafe_allow_html=True)

    df = load_data()

    tabs = st.tabs(["Distribution", "Default Patterns", "Correlations", "Payment Behavior"])

    # ── Tab 1: Distribution ──
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="AGE", nbins=30, color_discrete_sequence=[ACCENT])
            fig.update_traces(marker_line_width=0)
            chart_layout(fig, "Age Distribution")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            lim_bins = pd.cut(df["LIMIT_BAL"], bins=10).value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=[str(i) for i in lim_bins.index],
                y=lim_bins.values,
                marker_color=ACCENT2,
                marker_line_width=0,
            ))
            chart_layout(fig, "Credit Limit Distribution")
            fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            edu_map = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
            edu_cnt = df["EDUCATION"].map(edu_map).value_counts()
            fig = go.Figure(go.Bar(
                x=edu_cnt.index, y=edu_cnt.values,
                marker_color=[ACCENT, ACCENT2, WARNING, DANGER],
                marker_line_width=0,
            ))
            chart_layout(fig, "Education Level Counts")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c4:
            sex_map = {1: "Male", 2: "Female"}
            sex_cnt = df["SEX"].map(sex_map).value_counts()
            fig = go.Figure(go.Pie(
                labels=sex_cnt.index, values=sex_cnt.values,
                hole=0.55,
                marker_colors=[ACCENT, "#8b5cf6"],
                textinfo="percent+label",
            ))
            chart_layout(fig, "Gender Distribution", height=320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2: Default Patterns ──
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            # Default by age group
            df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[20,30,40,50,60,80],
                                     labels=["21-30","31-40","41-50","51-60","61+"])
            age_def = df.groupby("AGE_GROUP", observed=True)["DEFAULT"].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=age_def["AGE_GROUP"].astype(str),
                y=(age_def["DEFAULT"] * 100).round(1),
                marker_color=[ACCENT, ACCENT2, WARNING, DANGER, "#8b5cf6"],
                marker_line_width=0,
                text=(age_def["DEFAULT"]*100).round(1).astype(str)+"%",
                textposition="outside",
            ))
            chart_layout(fig, "Default Rate by Age Group")
            fig.update_yaxes(title_text="Default Rate (%)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            edu_map = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
            df["EDU_LABEL"] = df["EDUCATION"].map(edu_map)
            edu_def = df.groupby("EDU_LABEL")["DEFAULT"].mean().sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                x=(edu_def.values * 100).round(1),
                y=edu_def.index,
                orientation="h",
                marker_color=ACCENT,
                marker_line_width=0,
                text=(edu_def.values*100).round(1).astype(str)+"%",
                textposition="outside",
            ))
            chart_layout(fig, "Default Rate by Education")
            fig.update_xaxes(title_text="Default Rate (%)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            # Credit limit vs default
            df["LIMIT_GROUP"] = pd.cut(df["LIMIT_BAL"],
                bins=[0,50000,100000,200000,500000,1000001],
                labels=["<50K","50-100K","100-200K","200-500K","500K+"])
            lim_def = df.groupby("LIMIT_GROUP", observed=True)["DEFAULT"].mean().reset_index()
            fig = go.Figure(go.Scatter(
                x=lim_def["LIMIT_GROUP"].astype(str),
                y=(lim_def["DEFAULT"]*100).round(1),
                mode="lines+markers",
                line=dict(color=DANGER, width=2.5),
                marker=dict(size=8, color=DANGER),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.1)",
            ))
            chart_layout(fig, "Default Rate by Credit Limit")
            fig.update_yaxes(title_text="Default Rate (%)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c4:
            # Default count
            cnt = df["DEFAULT"].value_counts()
            fig = go.Figure(go.Bar(
                x=["No Default (0)", "Default (1)"],
                y=cnt.values,
                marker_color=[ACCENT2, DANGER],
                marker_line_width=0,
                text=cnt.values,
                textposition="outside",
            ))
            chart_layout(fig, "Default Class Distribution")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 3: Correlations ──
    with tabs[2]:
        numeric_cols = ["LIMIT_BAL","AGE","BILL_AMT1","BILL_AMT2","BILL_AMT3",
                        "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_0","PAY_2","DEFAULT"]
        corr = df[numeric_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0,"#1e3a5f"],[0.5,"#1e293b"],[1,"#3b82f6"]],
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "#e2e8f0"},
        ))
        chart_layout(fig, "Feature Correlation Heatmap", height=500)
        fig.update_layout(margin=dict(t=50, b=10, l=100, r=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 4: Payment Behavior ──
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            months   = ["Sep","Aug","Jul","Jun","May","Apr"]
            bill_avg = df.groupby("DEFAULT")[
                ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
            ].mean()
            fig = go.Figure()
            for label, clr in [(0, ACCENT2), (1, DANGER)]:
                fig.add_trace(go.Scatter(
                    x=months, y=bill_avg.loc[label].values,
                    name="No Default" if label == 0 else "Default",
                    line=dict(color=clr, width=2.5),
                    mode="lines+markers",
                    marker=dict(size=7),
                ))
            chart_layout(fig, "Average Bill Amount by Month")
            fig.update_yaxes(title_text="NT Dollar")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            pamt_avg = df.groupby("DEFAULT")[
                ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
            ].mean()
            fig = go.Figure()
            for label, clr in [(0, ACCENT2), (1, DANGER)]:
                fig.add_trace(go.Scatter(
                    x=months, y=pamt_avg.loc[label].values,
                    name="No Default" if label == 0 else "Default",
                    line=dict(color=clr, width=2.5, dash="solid"),
                    mode="lines+markers",
                    marker=dict(size=7),
                ))
            chart_layout(fig, "Average Payment Amount by Month")
            fig.update_yaxes(title_text="NT Dollar")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────
elif page == "🧠  Model Insights":
    st.markdown("<div class='page-heading'>🧠 Model Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Performance metrics, feature importance, and explainability for the Random Forest model.</div>", unsafe_allow_html=True)

    metrics = load_metrics()

    # KPI row
    kpis = [
        ("Accuracy",  f"{metrics['accuracy']*100:.2f}%",  "Overall correct predictions"),
        ("Precision", f"{metrics['precision']*100:.2f}%", "True positives / predicted positives"),
        ("Recall",    f"{metrics['recall']*100:.2f}%",    "Defaults correctly identified"),
        ("F1 Score",  f"{metrics['f1']*100:.2f}%",        "Balance of precision & recall"),
        ("ROC-AUC",   f"{metrics['auc']*100:.2f}%",       "Discrimination ability"),
    ]
    cols = st.columns(5)
    for col, (lbl, val, desc) in zip(cols, kpis):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='metric-val'>{val}</div>
            <div class='metric-lbl'>{lbl}</div>
            <div style='font-size:0.65rem; color:#334155; margin-top:0.3rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='section-tag'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        labels = ["No Default", "Default"]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: No Default", "Predicted: Default"],
            y=["Actual: No Default", "Actual: Default"],
            colorscale=[[0, "#0d1424"], [1, ACCENT]],
            showscale=False,
            text=cm, texttemplate="<b>%{text}</b>",
            textfont={"size": 22, "color": "white"},
        ))
        chart_layout(fig, height=320)
        fig.update_layout(margin=dict(t=20, b=20, l=120, r=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class='card' style='font-size:0.82rem; color:#94a3b8; line-height:1.8;'>
        <b style='color:#e2e8f0;'>Reading the matrix:</b><br>
        ✅ <b style='color:{ACCENT2};'>TN {cm[0][0]:,}</b> — correctly predicted no default<br>
        ✅ <b style='color:{ACCENT2};'>TP {cm[1][1]:,}</b> — correctly predicted default<br>
        ❌ <b style='color:{DANGER};'>FP {cm[0][1]:,}</b> — false alarms (said default, wasn't)<br>
        ❌ <b style='color:{WARNING};'>FN {cm[1][0]:,}</b> — missed defaults (most costly)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-tag'>Feature Importance (Top 15)</div>", unsafe_allow_html=True)
        fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=True).tail(15)
        colors = [ACCENT if "PAY" in i else ACCENT2 if "BILL" in i else WARNING if "PAY_AMT" in i else "#8b5cf6"
                  for i in fi.index]
        fig = go.Figure(go.Bar(
            x=fi.values,
            y=fi.index,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=(fi.values * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        chart_layout(fig, height=420)
        fig.update_xaxes(title_text="Importance Score")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>Model Architecture</div>", unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    arch_info = [
        ("🌳 Algorithm", "Random Forest Classifier"),
        ("🔢 Estimators", "200 decision trees"),
        ("📏 Max Depth", "10 levels per tree"),
        ("⚖️ Class Weight", "Balanced (handles imbalance)"),
        ("📐 Train/Test Split", "80% / 20%"),
        ("🎲 Random State", "42 (reproducible)"),
    ]
    for i, (k, v) in enumerate(arch_info):
        target_col = [a1, a2, a3][i % 3]
        target_col.markdown(f"""
        <div class='card' style='padding:1rem;'>
        <div style='font-size:0.72rem; color:#64748b; text-transform:uppercase;'>{k}</div>
        <div style='font-size:0.95rem; font-weight:600; color:#e2e8f0; margin-top:0.2rem;'>{v}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='margin-top:1rem; background:rgba(59,130,246,0.06); border-color:rgba(59,130,246,0.2);'>
    <b style='color:#e2e8f0;'>🔍 Non-Technical Explanation</b><br><br>
    <span style='color:#94a3b8; font-size:0.85rem; line-height:1.8;'>
    Think of the Random Forest as 200 credit analysts, each looking at the customer's data from a slightly different angle. 
    Each analyst votes: "default" or "no default." The majority vote wins. <br><br>
    The most important signal is <b style='color:#3b82f6;'>PAY_0</b> — whether the customer paid on time last month. 
    A customer who skipped September's payment is far more likely to default than one who paid in full. 
    This single feature explains nearly 25% of the model's decisions.
    </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: REPORT DOWNLOAD
# ─────────────────────────────────────────────
elif page == "📥  Report Download":
    st.markdown("<div class='page-heading'>📥 Report Download</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Download a formatted prediction report. Run a prediction first on the Prediction page.</div>", unsafe_allow_html=True)

    if "last_prediction" not in st.session_state:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem;'>
            <div style='font-size:2.5rem; margin-bottom:1rem;'>📋</div>
            <div style='font-size:1.1rem; color:#e2e8f0; font-weight:600; margin-bottom:0.5rem;'>No Prediction Found</div>
            <div style='color:#64748b; font-size:0.85rem;'>Please go to the <b>Prediction</b> page first and run a prediction.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        p = st.session_state["last_prediction"]
        inp = p["inputs"]
        prob = p["prob"]
        pred = p["pred"]
        conf = p["conf"]
        ts   = p["timestamp"]

        risk_pct = prob * 100
        if risk_pct >= 60:
            risk_level = "HIGH RISK"
        elif risk_pct >= 35:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        verdict = "LIKELY TO DEFAULT" if pred == 1 else "UNLIKELY TO DEFAULT"

        # Preview card
        st.markdown("<div class='section-tag'>Report Preview</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
            <div class='card'>
            <div style='font-family:JetBrains Mono, monospace; font-size:0.78rem; color:#94a3b8; line-height:2;'>
            <b style='color:#3b82f6; font-size:1rem;'>CREDITGUARD AI — PREDICTION REPORT</b><br>
            ═══════════════════════════════════════<br>
            Generated : {ts}<br>
            ───────────────────────────────────────<br>
            <b style='color:#e2e8f0;'>CUSTOMER PROFILE</b><br>
            Age            : {inp['AGE']}<br>
            Gender         : {p['sex_label']}<br>
            Education      : {p['edu_label']}<br>
            Marital Status : {p['mar_label']}<br>
            Credit Limit   : NT$ {inp['LIMIT_BAL']:,}<br>
            ───────────────────────────────────────<br>
            <b style='color:#e2e8f0;'>RECENT PAYMENT STATUS</b><br>
            September : {inp['PAY_0']}  |  August : {inp['PAY_2']}<br>
            July      : {inp['PAY_3']}  |  June   : {inp['PAY_4']}<br>
            ───────────────────────────────────────<br>
            <b style='color:#e2e8f0;'>PREDICTION RESULT</b><br>
            Verdict         : {verdict}<br>
            Default Prob.   : {risk_pct:.1f}%<br>
            Model Confidence: {conf:.1f}%<br>
            Risk Level      : {risk_level}<br>
            ═══════════════════════════════════════
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            recs_default = [
                "Reduce credit limit immediately",
                "Initiate contact for payment plan",
                "Flag for enhanced monitoring",
                "Review credit terms & conditions",
            ]
            recs_safe = [
                "Consider upsell opportunities",
                "Offer loyalty/reward programs",
                "Standard 90-day re-assessment",
                "Maintain current credit terms",
            ]
            recs = recs_default if pred == 1 else recs_safe

            st.markdown(f"""
            <div class='{"card-danger" if pred==1 else "card-success"}' style='height:100%;'>
            <b style='color:#e2e8f0;'>Recommendations</b><br><br>
            {"".join(f"<div style='margin:0.5rem 0; font-size:0.82rem; color:#94a3b8;'>→ {r}</div>" for r in recs)}
            </div>
            """, unsafe_allow_html=True)

        # Generate downloadable text report
        pay_status_desc = {-2: "No consumption", -1: "Paid fully", 0: "Revolving credit used",
                           1: "1 month delay", 2: "2 month delay", 3: "3 month delay",
                           4: "4 month delay", 5: "5 month delay", 6: "6+ month delay"}

        report_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║              CREDITGUARD AI — PREDICTION REPORT                  ║
║              Built by Neha Tiwari | github.com/tiwarineha73     ║
╚══════════════════════════════════════════════════════════════════╝

Report Generated : {ts}
Model            : Random Forest Classifier (200 estimators)
Dataset          : UCI Credit Card Default (Taiwan, 2005)

════════════════════════════════════════════════════════════════════
  CUSTOMER PROFILE
════════════════════════════════════════════════════════════════════

  Age              : {inp['AGE']} years
  Gender           : {p['sex_label']}
  Education        : {p['edu_label']}
  Marital Status   : {p['mar_label']}
  Credit Limit     : NT$ {inp['LIMIT_BAL']:,}

════════════════════════════════════════════════════════════════════
  PAYMENT HISTORY (Repayment Status)
════════════════════════════════════════════════════════════════════

  September : {inp['PAY_0']} ({pay_status_desc.get(inp['PAY_0'], 'Delayed')})
  August    : {inp['PAY_2']} ({pay_status_desc.get(inp['PAY_2'], 'Delayed')})
  July      : {inp['PAY_3']} ({pay_status_desc.get(inp['PAY_3'], 'Delayed')})
  June      : {inp['PAY_4']} ({pay_status_desc.get(inp['PAY_4'], 'Delayed')})
  May       : {inp['PAY_5']} ({pay_status_desc.get(inp['PAY_5'], 'Delayed')})
  April     : {inp['PAY_6']} ({pay_status_desc.get(inp['PAY_6'], 'Delayed')})

════════════════════════════════════════════════════════════════════
  BILL AMOUNTS (NT Dollar)
════════════════════════════════════════════════════════════════════

  September : NT$ {inp['BILL_AMT1']:,}
  August    : NT$ {inp['BILL_AMT2']:,}
  July      : NT$ {inp['BILL_AMT3']:,}
  June      : NT$ {inp['BILL_AMT4']:,}
  May       : NT$ {inp['BILL_AMT5']:,}
  April     : NT$ {inp['BILL_AMT6']:,}

════════════════════════════════════════════════════════════════════
  PAYMENT AMOUNTS (NT Dollar)
════════════════════════════════════════════════════════════════════

  September : NT$ {inp['PAY_AMT1']:,}
  August    : NT$ {inp['PAY_AMT2']:,}
  July      : NT$ {inp['PAY_AMT3']:,}
  June      : NT$ {inp['PAY_AMT4']:,}
  May       : NT$ {inp['PAY_AMT5']:,}
  April     : NT$ {inp['PAY_AMT6']:,}

════════════════════════════════════════════════════════════════════
  PREDICTION RESULT
════════════════════════════════════════════════════════════════════

  ★  VERDICT          : {verdict}
  ★  Default Prob.    : {risk_pct:.1f}%
  ★  Model Confidence : {conf:.1f}%
  ★  Risk Level       : {risk_level}

════════════════════════════════════════════════════════════════════
  RECOMMENDATIONS
════════════════════════════════════════════════════════════════════

{"".join(f"  → {r}" + chr(10) for r in (recs_default if pred == 1 else recs_safe))}
════════════════════════════════════════════════════════════════════
  DISCLAIMER
════════════════════════════════════════════════════════════════════

  This report is generated for portfolio/educational purposes.
  The model is trained on historical data and may not generalize
  to all real-world scenarios. Do not use for actual financial
  or credit decisions without professional review.

════════════════════════════════════════════════════════════════════
  MODEL PERFORMANCE (Test Set)
════════════════════════════════════════════════════════════════════

  Accuracy  : {load_metrics()['accuracy']*100:.2f}%
  Precision : {load_metrics()['precision']*100:.2f}%
  Recall    : {load_metrics()['recall']*100:.2f}%
  F1 Score  : {load_metrics()['f1']*100:.2f}%
  ROC-AUC   : {load_metrics()['auc']*100:.2f}%

══════════════════════════════════════════════════════════════════
  CreditGuard AI | Neha Tiwari | github.com/tiwarineha73
══════════════════════════════════════════════════════════════════
"""

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        filename = f"creditguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button(
            label="📥 Download Full Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
        )

        st.markdown("""
        <div style='color:#475569; font-size:0.72rem; margin-top:0.5rem;'>
        The report includes all input fields, prediction outcome, risk level, and model recommendations.
        Open in any text editor for a clean formatted view.
        </div>
        """, unsafe_allow_html=True)
