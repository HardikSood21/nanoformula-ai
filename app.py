"""
NanoFormula AI — Multi-Polymer Nanoparticle Formulation Optimizer
PLGA + Chitosan-TPP Models
Developed by: Hardik Sood | IIT (BHU) Varanasi
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from datetime import datetime
import random

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be first
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NanoFormula AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=Space+Mono:wght@400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: #04080f;
    background-image:
        radial-gradient(ellipse 90% 55% at 15% -5%,  rgba(0,210,140,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 70% 45% at 85% 105%, rgba(0,130,255,0.07) 0%, transparent 55%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 1300px !important;
}

/* ── Sidebar – FULLY VISIBLE ── */
[data-testid="stSidebar"] {
    background: #060c18 !important;
    border-right: 1px solid rgba(0,210,140,0.12) !important;
    min-width: 310px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(190, 220, 250, 0.82) !important;
    font-size: 0.84rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e8f4ff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-top: 1.2rem !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8f4ff !important;
    font-size: 0.84rem !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div div div {
    background: linear-gradient(90deg, #00d28c, #00aaff) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07) !important; margin: 0.8rem 0 !important; }

/* ── Hero ── */
.hero {
    padding: 3rem 2rem 2rem;
    text-align: center;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem; letter-spacing: 4px; text-transform: uppercase;
    color: rgba(0,210,140,0.65); margin-bottom: 1.1rem;
    display: flex; align-items: center; justify-content: center; gap: 12px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: ''; width: 48px; height: 1px; background: rgba(0,210,140,0.25);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6.5vw, 5.2rem);
    font-weight: 800; line-height: 0.93; letter-spacing: -3px;
    background: linear-gradient(150deg, #ffffff 15%, #00d28c 55%, #00aaff 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1rem; font-weight: 300; font-style: italic;
    color: rgba(170,205,240,0.5);
    max-width: 540px; margin: 0 auto 2.2rem; line-height: 1.8;
}
.stat-bar {
    display: flex; justify-content: center; flex-wrap: wrap; gap: 0;
    max-width: 780px; margin: 0 auto;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; overflow: hidden;
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(12px);
}
.stat-item {
    flex: 1; min-width: 115px; padding: 1.1rem 0.8rem;
    text-align: center; border-right: 1px solid rgba(255,255,255,0.06);
}
.stat-item:last-child { border-right: none; }
.stat-val {
    font-family: 'Syne', sans-serif; font-size: 1.55rem; font-weight: 700;
    color: #00d28c; line-height: 1;
}
.stat-lbl {
    font-family: 'Space Mono', monospace; font-size: 0.62rem;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: rgba(140,180,220,0.38); margin-top: 5px;
}

/* ── Polymer Switcher ── */
.poly-switcher {
    display: flex; gap: 10px; justify-content: center;
    margin: 2rem 0 0.5rem; flex-wrap: wrap;
}
.poly-btn {
    padding: 10px 28px; border-radius: 50px;
    font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.5px; cursor: pointer; transition: all 0.25s; border: none;
}
.poly-btn-plga {
    background: rgba(0,210,140,0.12);
    border: 1.5px solid rgba(0,210,140,0.4);
    color: #00d28c;
}
.poly-btn-plga.active {
    background: linear-gradient(135deg,#00d28c,#00c4b8);
    color: #03070f; border-color: transparent;
    box-shadow: 0 4px 20px rgba(0,210,140,0.35);
}
.poly-btn-cs {
    background: rgba(0,130,255,0.1);
    border: 1.5px solid rgba(0,130,255,0.35);
    color: #0096ff;
}
.poly-btn-cs.active {
    background: linear-gradient(135deg,#0096ff,#7b61ff);
    color: #ffffff; border-color: transparent;
    box-shadow: 0 4px 20px rgba(0,130,255,0.3);
}

/* ── Section title ── */
.sec-title {
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    color: rgba(210,235,255,0.85); letter-spacing: -0.3px;
    padding-bottom: 9px; border-bottom: 1px solid rgba(255,255,255,0.06);
    margin: 1.8rem 0 1.1rem;
    display: flex; align-items: center; gap: 8px;
}
.sec-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink:0; }
.dot-green { background: #00d28c; }
.dot-blue  { background: #0096ff; }

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.4rem; margin-bottom: 1rem;
}

/* ── KPI strip ── */
.kpi-strip {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 1.1rem 0;
}
.kpi { background: rgba(0,210,140,0.05); border: 1px solid rgba(0,210,140,0.14);
    border-radius: 14px; padding: 1.1rem 0.8rem; text-align: center; }
.kpi-val { font-family: 'Space Mono',monospace; font-size: 1.5rem; font-weight:700;
    color: #00d28c; line-height:1; }
.kpi-unit { font-size:0.72rem; color:rgba(0,210,140,0.5); margin-left:2px; }
.kpi-lbl { font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:1.5px;
    text-transform:uppercase; color:rgba(140,180,220,0.38); margin-top:5px; }

/* ── Alert boxes ── */
.abox { border-radius: 0 10px 10px 0; padding: 11px 16px;
    font-size: 0.875rem; line-height: 1.65; margin: 10px 0; }
.abox-info    { background:rgba(0,150,255,0.07); border-left:3px solid #0096ff; color:rgba(180,220,255,0.82); }
.abox-success { background:rgba(0,210,140,0.07); border-left:3px solid #00d28c; color:rgba(160,255,210,0.85); }
.abox-warn    { background:rgba(255,185,0,0.07);  border-left:3px solid #ffb900; color:rgba(255,220,120,0.85); }

/* ── Protocol card ── */
.proto { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:1.5rem 1.8rem; }
.proto-head { font-family:'Syne',sans-serif; font-size:0.92rem; font-weight:700;
    color:#00d28c; margin-bottom:1.1rem; }
.pstep { display:flex; gap:13px; margin-bottom:11px; align-items:flex-start; }
.pnum { width:25px; height:25px; background:rgba(0,210,140,0.1);
    border:1px solid rgba(0,210,140,0.22); color:#00d28c; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.7rem; font-weight:700; flex-shrink:0; font-family:'Space Mono',monospace; }
.ptxt { font-size:0.86rem; color:rgba(185,218,245,0.74); line-height:1.65; }

/* ── Mini formulation card ── */
.fcard { background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:12px 14px; margin-bottom:9px; }
.fcard-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:7px; }
.fcard-rank { font-family:'Syne',sans-serif; font-weight:700; color:#e8f4ff; font-size:0.9rem; }
.fcard-scores { font-family:'Space Mono',monospace; font-size:0.73rem; color:#00d28c; }
.fcard-params { font-size:0.71rem; color:rgba(145,180,220,0.48); line-height:1.9; }

/* ── Feature importance ── */
.firow { display:flex; align-items:center; gap:10px; padding:6px 0;
    border-bottom:1px solid rgba(255,255,255,0.04); }
.finame { font-size:0.79rem; color:rgba(175,210,240,0.65); min-width:195px; }
.fibarbg { flex:1; height:5px; background:rgba(255,255,255,0.06); border-radius:3px; }
.fibarfg { height:100%; border-radius:3px; background:linear-gradient(90deg,#00d28c,#00aaff); }
.fipct { font-size:0.73rem; color:rgba(0,210,140,0.65); font-family:'Space Mono',monospace;
    min-width:36px; text-align:right; }

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg,#00d28c,#00aaff) !important;
    color: #03070f !important; font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:0.88rem !important;
    letter-spacing:1px !important; border:none !important;
    border-radius:10px !important; padding:0.65rem 1.4rem !important;
    width:100% !important; transition:all 0.25s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 26px rgba(0,210,140,0.32) !important;
}
.stDownloadButton > button {
    background:rgba(0,210,140,0.08) !important; color:#00d28c !important;
    border:1px solid rgba(0,210,140,0.3) !important; border-radius:10px !important;
    font-weight:600 !important; width:auto !important;
}
[data-testid="stMetricValue"] {
    font-family:'Space Mono',monospace !important; color:#00d28c !important; font-size:1.5rem !important;
}
[data-testid="stMetricLabel"] {
    font-size:0.68rem !important; color:rgba(145,180,215,0.42) !important;
    text-transform:uppercase !important; letter-spacing:1.2px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.025) !important; border-radius:12px !important;
    padding:4px !important; gap:2px !important;
}
.stTabs [data-baseweb="tab"] {
    color:rgba(155,195,230,0.48) !important; font-weight:500 !important;
    border-radius:9px !important; font-size:0.84rem !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(0,210,140,0.12) !important; color:#00d28c !important;
}
[data-testid="stDataFrame"] { border-radius:12px !important; overflow:hidden; }
div[data-testid="stDataFrame"] > div { border-radius:12px !important; }
.stSpinner > div { border-top-color:#00d28c !important; }

/* ── Footer ── */
.footer {
    border-top:1px solid rgba(255,255,255,0.05); margin-top:4rem;
    padding:1.5rem 0 0.5rem; text-align:center;
    font-family:'Space Mono',monospace; font-size:0.65rem;
    letter-spacing:1.2px; color:rgba(110,155,200,0.22);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#080f1c', 'axes.facecolor': '#0b1525',
    'axes.edgecolor': '#18304a',   'axes.labelcolor': '#78a8cc',
    'xtick.color': '#486880',      'ytick.color': '#486880',
    'text.color': '#aed0f0',       'grid.color': '#18304a',
    'grid.alpha': 0.5,             'font.family': 'sans-serif',
    'axes.spines.top': False,      'axes.spines.right': False,
})

# ══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    m = {}
    # PLGA
    try:
        with open('model_particle_size_final.pkl','rb') as f: m['plga_size'] = pickle.load(f)
        with open('model_ee_final.pkl','rb') as f:           m['plga_ee']   = pickle.load(f)
        m['plga_data']   = pd.read_csv('PLGA_nanoparticles_dataset.csv')
        m['plga_loaded'] = True
    except Exception as e:
        m['plga_loaded'] = False; m['plga_error'] = str(e)
    # Chitosan
    try:
        with open('model_chitosan_size.pkl','rb') as f:  m['cs_size']     = pickle.load(f)
        with open('chitosan_features.pkl','rb') as f:    m['cs_features'] = pickle.load(f)
        m['cs_data']   = pd.read_csv('chitosan_nanoparticles_dataset.csv')
        m['cs_loaded'] = True
    except Exception as e:
        m['cs_loaded'] = False; m['cs_error'] = str(e)
    return m

models = load_models()

# ══════════════════════════════════════════════════════════════════
# SESSION STATE — which polymer tab is active
# ══════════════════════════════════════════════════════════════════
if 'polymer' not in st.session_state:
    st.session_state.polymer = 'PLGA'

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Machine Learning · Drug Delivery · Nanomedicine</div>
  <div class="hero-title">NanoFormula<br>AI</div>
  <div class="hero-sub">
    AI-powered virtual screening for PLGA &amp; Chitosan-TPP nanoparticle formulations.<br>
    Predict particle size &amp; encapsulation efficiency — replace weeks of trial-and-error.
  </div>
  <div class="stat-bar">
    <div class="stat-item"><div class="stat-val">433</div><div class="stat-lbl">PLGA Sets</div></div>
    <div class="stat-item"><div class="stat-val">0.88</div><div class="stat-lbl">PLGA R²</div></div>
    <div class="stat-item"><div class="stat-val">44</div><div class="stat-lbl">CS Sets</div></div>
    <div class="stat-item"><div class="stat-val">0.83</div><div class="stat-lbl">CS R²</div></div>
    <div class="stat-item"><div class="stat-val">20K</div><div class="stat-lbl">Candidates</div></div>
    <div class="stat-item"><div class="stat-val">~5</div><div class="stat-lbl">Trials Needed</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# POLYMER SELECTION BUTTONS
# ══════════════════════════════════════════════════════════════════
col_l, col_plga, col_cs, col_r = st.columns([2, 1, 1, 2])
with col_plga:
    if st.button("🧪  PLGA", key="btn_plga"):
        st.session_state.polymer = 'PLGA'
with col_cs:
    if st.button("🔬  Chitosan-TPP", key="btn_cs"):
        st.session_state.polymer = 'Chitosan'

polymer = st.session_state.polymer
dot_cls = "dot-green" if polymer == "PLGA" else "dot-blue"
accent  = "#00d28c"   if polymer == "PLGA" else "#0096ff"

# Active indicator
st.markdown(f"""
<div style="text-align:center;margin-bottom:0.5rem">
  <span style="font-family:'Space Mono',monospace;font-size:0.68rem;letter-spacing:2px;
               color:{accent};text-transform:uppercase">
    ▶ {polymer} Mode Active
  </span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR — content changes based on active polymer
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown(f"""
<div style="padding:8px 0 4px">
  <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8f4ff">
    🧬 NanoFormula AI
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
              color:{accent};letter-spacing:2px;margin-top:3px">
    {polymer} OPTIMIZER · v4.0
  </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

run = False  # default

# ─────────────────────────────────────────────
# PLGA SIDEBAR
# ─────────────────────────────────────────────
if polymer == 'PLGA':
    if not models.get('plga_loaded'):
        st.sidebar.error(f"PLGA model not found.\n{models.get('plga_error','')}")
    else:
        st.sidebar.markdown("### Drug Properties")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            mol_MW   = st.number_input("MW (g/mol)",      100.0, 1000.0, 420.5, 10.0)
            mol_logP = st.number_input("LogP",            -5.0,  10.0,   3.1,   0.1)
            mol_TPSA = st.number_input("TPSA (Ų)",        0.0,   300.0,  72.0,  5.0)
            mol_mp   = st.number_input("Melting Pt (°C)", 0.0,   500.0,  175.0, 5.0)
        with c2:
            mol_Hacc = st.number_input("H-Acceptors",     0, 20, 5, 1)
            mol_Hdon = st.number_input("H-Donors",        0, 10, 2, 1)
            mol_het  = st.number_input("Heteroatoms",     0, 30, 7, 1)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### PLGA Grade")
        poly_mode = st.sidebar.radio("Selection:", ["Auto-optimize", "Specify grade"],
                                     label_visibility="collapsed")
        polymer_constraints = None
        if poly_mode == "Specify grade":
            df_ = models['plga_data']
            sel_mw   = st.sidebar.selectbox("PLGA MW (kDa)", sorted(df_['polymer_MW'].unique()))
            sel_laga = st.sidebar.selectbox("LA/GA Ratio",   sorted(df_['LA/GA'].unique()))
            polymer_constraints = {'polymer_MW': sel_mw, 'LA/GA': sel_laga}

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Targets")
        target_size  = st.sidebar.slider("Target Size (nm)",      50, 300, 180, 5)
        min_ee       = st.sidebar.slider("Minimum EE%",           40, 95,  70,  5)
        n_recs       = st.sidebar.slider("No. of Recommendations", 3, 10,  5,   1)
        size_weight  = st.sidebar.slider("Priority: Size ← → EE", 0.3, 0.9, 0.7, 0.05,
                                          help="0.7 = 70% weight on size, 30% on EE")
        st.sidebar.markdown("---")
        run = st.sidebar.button("▶  Run PLGA Optimization", key="run_plga")

# ─────────────────────────────────────────────
# CHITOSAN SIDEBAR
# ─────────────────────────────────────────────
else:
    if not models.get('cs_loaded'):
        st.sidebar.error(f"Chitosan model not found.\n{models.get('cs_error','')}")
    else:
        st.sidebar.markdown("### Chitosan Grade")
        cs_mode = st.sidebar.radio("Selection:", ["Auto-optimize", "Specify grade"],
                                    label_visibility="collapsed")
        fix_mw     = False
        chitosan_mw = None
        if cs_mode == "Specify grade":
            chitosan_mw = st.sidebar.selectbox(
                "Chitosan MW",
                [5, 20, 50, 310],
                format_func=lambda x: f"{x} kDa {'(HMW)' if x==310 else '(LMW)' if x==50 else ''}"
            )
            fix_mw = True

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Targets")
        target_size_cs = st.sidebar.slider("Target Size (nm)",       50, 250, 150, 5,  key="cs_sz")
        n_recs_cs      = st.sidebar.slider("No. of Recommendations",  3,  10,  5,  1,  key="cs_nr")
        st.sidebar.markdown("---")
        run = st.sidebar.button("▶  Run Chitosan Optimization", key="run_cs")

# ══════════════════════════════════════════════════════════════════
# PRE-RUN INFO VIEW
# ══════════════════════════════════════════════════════════════════
if not run:
    st.markdown(f'<div class="sec-title"><span class="sec-dot {dot_cls}"></span>Model Overview</div>',
                unsafe_allow_html=True)

    if polymer == 'PLGA':
        c1, c2, c3 = st.columns(3)
        cards = [
            ("Particle Size Model", "#00d28c",
             [("Algorithm","Random Forest"),("R² Score","0.88"),
              ("MAE","±22 nm"),("Training Samples","433"),("Input Features","15")]),
            ("EE% Model", "#00aaff",
             [("Algorithm","Random Forest"),("R² Score","0.47"),
              ("Training Samples","433"),("Drug Classes","65"),("Output Range","0–100%")]),
            ("Virtual Screening", "#a78bfa",
             [("Candidates / Run","20,000"),("Method","Monte Carlo"),
              ("Diversity Filter","Yes"),("Scoring","Multi-objective"),("Output","Top-N Formulations")]),
        ]
        for col, (title, color, rows) in zip([c1, c2, c3], cards):
            rows_html = "".join([
                f'<div class="firow"><span class="finame">{k}</span>'
                f'<span style="font-size:0.79rem;color:{color}">{v}</span></div>'
                for k, v in rows
            ])
            col.markdown(f"""<div class="glass">
                <div style="font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:700;
                            color:#e8f4ff;margin-bottom:10px">{title}</div>
                {rows_html}</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-title"><span class="sec-dot {dot_cls}"></span>Feature Importance (Relative)</div>',
                    unsafe_allow_html=True)
        features = [
            ("Polymer MW (kDa)", 92), ("LA/GA Ratio", 88), ("Drug/Polymer Ratio", 85),
            ("Surfactant Concentration", 80), ("Surfactant HLB", 72), ("Drug LogP", 68),
            ("Aqueous/Organic Ratio", 65), ("Drug Molecular Weight", 60),
            ("pH", 54), ("Solvent Polarity Index", 48),
        ]
        fi_html = "".join([
            f'<div class="firow"><span class="finame">{n}</span>'
            f'<div class="fibarbg"><div class="fibarfg" style="width:{p}%"></div></div>'
            f'<span class="fipct">{p}%</span></div>'
            for n, p in features
        ])
        st.markdown(f'<div class="glass">{fi_html}</div>', unsafe_allow_html=True)

    else:  # Chitosan pre-run
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="glass">
                <div style="font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:700;color:#e8f4ff;margin-bottom:10px">Chitosan-TPP Model</div>
                <div class="firow"><span class="finame">Algorithm</span><span style="font-size:0.79rem;color:#0096ff">Random Forest</span></div>
                <div class="firow"><span class="finame">R² Score</span><span style="font-size:0.79rem;color:#0096ff">0.83</span></div>
                <div class="firow"><span class="finame">MAE</span><span style="font-size:0.79rem;color:#0096ff">±15–20 nm</span></div>
                <div class="firow"><span class="finame">Training Samples</span><span style="font-size:0.79rem;color:#0096ff">44</span></div>
                <div class="firow"><span class="finame">MW Grades</span><span style="font-size:0.79rem;color:#0096ff">5, 20, 50, 310 kDa</span></div>
                <div class="firow"><span class="finame">Predicts</span><span style="font-size:0.79rem;color:#0096ff">Particle Size only</span></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="glass">
                <div style="font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:700;color:#e8f4ff;margin-bottom:10px">Screening Parameters</div>
                <div class="firow"><span class="finame">Candidates / Run</span><span style="font-size:0.79rem;color:#a78bfa">10,000</span></div>
                <div class="firow"><span class="finame">CS Conc. Range</span><span style="font-size:0.79rem;color:#a78bfa">0.1 – 1.0 mg/mL</span></div>
                <div class="firow"><span class="finame">TPP Conc. Range</span><span style="font-size:0.79rem;color:#a78bfa">0.1 – 1.0 mg/mL</span></div>
                <div class="firow"><span class="finame">CS:TPP Ratio</span><span style="font-size:0.79rem;color:#a78bfa">Derived</span></div>
                <div class="firow"><span class="finame">Method</span><span style="font-size:0.79rem;color:#a78bfa">Ionic Gelation</span></div>
                <div class="firow"><span class="finame">Diversity Filter</span><span style="font-size:0.79rem;color:#a78bfa">By MW Grade</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="abox abox-warn">
            <strong>Note:</strong> The Chitosan-TPP model predicts <strong>particle size only</strong>.
            EE% data was not available for this polymer system.
            Trained on 44 blank nanoparticle formulations across 4 molecular weight grades.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="abox abox-info">
        <strong>How to use:</strong> {'Enter drug molecular properties in the sidebar, set your size and EE% targets, then click Run PLGA Optimization.' if polymer == 'PLGA' else 'Select chitosan grade (or auto-optimize), set your target size, then click Run Chitosan Optimization.'}
        The model will screen {'20,000' if polymer == 'PLGA' else '10,000'} candidate formulations and surface the top-ranked recommendations.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ───────────────────  PLGA OPTIMIZATION  ──────────────────────────
# ══════════════════════════════════════════════════════════════════
if run and polymer == 'PLGA' and models.get('plga_loaded'):

    drug_props = {
        'mol_MW': mol_MW, 'mol_logP': mol_logP, 'mol_TPSA': mol_TPSA,
        'mol_melting_point': mol_mp, 'mol_Hacceptors': mol_Hacc,
        'mol_Hdonors': mol_Hdon, 'mol_heteroatoms': mol_het
    }

    with st.spinner("Screening 20,000 PLGA formulation candidates…"):
        df_p       = models['plga_data']
        successful = df_p[df_p['particle_size'] < target_size]
        if len(successful) < 20:
            successful = df_p[df_p['particle_size'] < 300]

        space = {
            'polymer_MW':             successful['polymer_MW'].unique().tolist(),
            'LA/GA':                  successful['LA/GA'].unique().tolist(),
            'drug/polymer':           np.linspace(successful['drug/polymer'].quantile(0.05),
                                                  successful['drug/polymer'].quantile(0.95), 50).tolist(),
            'surfactant_concentration': np.linspace(successful['surfactant_concentration'].min(),
                                                    successful['surfactant_concentration'].quantile(0.95), 30).tolist(),
            'surfactant_HLB':         successful['surfactant_HLB'].unique().tolist(),
            'aqueous/organic':        successful['aqueous/organic'].unique().tolist(),
            'pH':                     successful['pH'].unique().tolist(),
            'solvent_polarity_index': successful['solvent_polarity_index'].unique().tolist(),
        }
        if polymer_constraints:
            for k, v in polymer_constraints.items():
                space[k] = [v]

        random.seed(42); np.random.seed(42)
        cands = [{
            **drug_props,
            'polymer_MW':             random.choice(space['polymer_MW']),
            'LA/GA':                  random.choice(space['LA/GA']),
            'drug/polymer':           random.choice(space['drug/polymer']),
            'surfactant_concentration': random.choice(space['surfactant_concentration']),
            'surfactant_HLB':         random.choice(space['surfactant_HLB']),
            'aqueous/organic':        random.choice(space['aqueous/organic']),
            'pH':                     random.choice(space['pH']),
            'solvent_polarity_index': random.choice(space['solvent_polarity_index']),
        } for _ in range(20000)]

        FEAT = ['polymer_MW','LA/GA','mol_MW','mol_logP','mol_TPSA','mol_melting_point',
                'mol_Hacceptors','mol_Hdonors','mol_heteroatoms','drug/polymer',
                'surfactant_concentration','surfactant_HLB','aqueous/organic',
                'pH','solvent_polarity_index']
        cdf = pd.DataFrame(cands)
        cdf['pred_size'] = models['plga_size'].predict(cdf[FEAT])
        cdf['pred_EE']   = np.clip(models['plga_ee'].predict(cdf[FEAT]), 0, 100)

        ee_w = 1 - size_weight
        cdf['size_score'] = np.where(cdf['pred_size'] <= target_size,
                                     (target_size - cdf['pred_size']) / target_size, -0.5)
        cdf['ee_score']   = (cdf['pred_EE'] - 50) / 50
        cdf['score']      = size_weight * cdf['size_score'] + ee_w * cdf['ee_score']

        top = cdf.nlargest(n_recs * 5, 'score')
        diverse, used = [], []
        for _, row in top.iterrows():
            if not any(abs(row['drug/polymer'] - u) < 0.015 for u in used):
                diverse.append(row); used.append(row['drug/polymer'])
            if len(diverse) >= n_recs: break
        if len(diverse) < n_recs:
            diverse = [r for _, r in top.head(n_recs).iterrows()]
        recs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)

    hit = (cdf['pred_size'] < target_size).sum()

    st.markdown('<div class="abox abox-success">✅ &nbsp; PLGA optimization complete — 20,000 candidates screened.</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi"><div class="kpi-val">20,000</div><div class="kpi-lbl">Screened</div></div>
      <div class="kpi"><div class="kpi-val">{hit:,}</div><div class="kpi-lbl">Within Target</div></div>
      <div class="kpi"><div class="kpi-val">{recs['pred_size'].min():.1f}<span class="kpi-unit">nm</span></div><div class="kpi-lbl">Best Size</div></div>
      <div class="kpi"><div class="kpi-val">{recs['pred_EE'].max():.1f}<span class="kpi-unit">%</span></div><div class="kpi-lbl">Best EE%</div></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋  Recommendations", "📊  Visual Analysis",
        "🔭  Search Space",     "🧪  Lab Protocol"
    ])

    # ── Tab 1: Table ──
    with tab1:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-green"></span>Top Recommended Formulations</div>',
                    unsafe_allow_html=True)
        disp = recs[['drug/polymer','polymer_MW','LA/GA','surfactant_concentration',
                      'surfactant_HLB','aqueous/organic','pH','pred_size','pred_EE','score']].copy()
        disp.insert(0, 'Rank', [f'F{i+1}' for i in range(len(disp))])
        disp.columns = ['Rank','Drug/Polymer','PLGA MW (kDa)','LA/GA','Surf. Conc (%)','Surf. HLB',
                         'Aq/Org','pH','Pred. Size (nm)','Pred. EE (%)','Score']
        st.dataframe(
            disp.style
                .format({'Drug/Polymer':'{:.4f}','PLGA MW (kDa)':'{:.1f}','LA/GA':'{:.2f}',
                         'Surf. Conc (%)':'{:.3f}','Surf. HLB':'{:.1f}','Aq/Org':'{:.2f}',
                         'pH':'{:.2f}','Pred. Size (nm)':'{:.1f}','Pred. EE (%)':'{:.1f}','Score':'{:.4f}'})
                .background_gradient(subset=['Pred. Size (nm)'], cmap='RdYlGn_r', vmin=50, vmax=300)
                .background_gradient(subset=['Pred. EE (%)'],    cmap='RdYlGn',   vmin=40, vmax=100)
                .background_gradient(subset=['Score'],           cmap='Blues'),
            use_container_width=True, height=min(380, 60 + len(disp)*52)
        )
        st.markdown("""<div class="abox abox-info">
            <strong>Score</strong> = weighted multi-objective (size + EE). Higher = better.
            Diversity filter ensures varied drug/polymer ratios across recommendations.
        </div>""", unsafe_allow_html=True)
        st.download_button("📥  Download CSV", disp.to_csv(index=False),
            f"plga_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    # ── Tab 2: Charts ──
    with tab2:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-green"></span>Visual Analysis</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            norm = plt.Normalize(recs['pred_EE'].min(), recs['pred_EE'].max())
            ax.barh(range(len(recs)), recs['pred_size'],
                    color=plt.cm.cool(norm(recs['pred_EE'])), height=0.62, edgecolor='none')
            ax.axvline(target_size, color='#ff6b6b', lw=1.5, ls='--', alpha=0.85,
                       label=f'Target: {target_size} nm')
            ax.set_yticks(range(len(recs)))
            ax.set_yticklabels([f'F{i+1}' for i in range(len(recs))], fontsize=9)
            ax.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax.set_title('Ranked Formulations — Particle Size', fontsize=10, fontweight='bold', pad=12)
            ax.legend(fontsize=8); ax.grid(axis='x', lw=0.5)
            for i, v in enumerate(recs['pred_size']):
                ax.text(v+1.5, i, f'{v:.1f}', va='center', fontsize=8, color='#8ab8d8')
            sm = cm.ScalarMappable(cmap='cool', norm=norm); sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
            cb.set_label('EE%', fontsize=8); cb.ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(7, 4.8))
            sc = ax2.scatter(recs['pred_size'], recs['pred_EE'], s=340,
                             c=recs['score'], cmap='plasma',
                             edgecolors='#3a6a9a', lw=1.2, alpha=0.92, zorder=3)
            ax2.axvspan(0, target_size, alpha=0.04, color='#00d28c')
            ax2.axhspan(min_ee, 100,    alpha=0.04, color='#00aaff')
            ax2.axvline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.6)
            ax2.axhline(min_ee,      color='#4a9eff', lw=1.2, ls='--', alpha=0.6)
            for i in range(len(recs)):
                ax2.annotate(f'F{i+1}', (recs.iloc[i]['pred_size'], recs.iloc[i]['pred_EE']),
                             fontsize=9, fontweight='bold', ha='center', va='center', color='white')
            ax2.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax2.set_ylabel('Predicted EE%', fontsize=9)
            ax2.set_title('Pareto Space: Size vs. EE%', fontsize=10, fontweight='bold', pad=12)
            ax2.grid(lw=0.5)
            cb2 = fig2.colorbar(sc, ax=ax2, pad=0.01, fraction=0.025)
            cb2.set_label('Score', fontsize=8); cb2.ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig2)

        c3, c4 = st.columns(2)
        with c3:
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sc3 = ax3.scatter(recs['drug/polymer'], recs['pred_size'], s=230,
                              c=recs['pred_EE'], cmap='YlGn',
                              edgecolors='#2a4a6a', lw=1, alpha=0.9)
            ax3.axhline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.7)
            for i in range(len(recs)):
                ax3.annotate(f'F{i+1}', (recs.iloc[i]['drug/polymer'], recs.iloc[i]['pred_size']),
                             textcoords='offset points', xytext=(6,4), fontsize=8, color='#8ab0d0')
            ax3.set_xlabel('Drug/Polymer Ratio', fontsize=9)
            ax3.set_ylabel('Predicted Size (nm)', fontsize=9)
            ax3.set_title('Drug/Polymer Ratio vs. Particle Size', fontsize=10, fontweight='bold', pad=12)
            ax3.grid(lw=0.5)
            fig3.colorbar(sc3, ax=ax3, pad=0.01, fraction=0.025).set_label('EE%', fontsize=8)
            plt.tight_layout(); st.pyplot(fig3)

        with c4:
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            sc4 = ax4.scatter(recs['surfactant_concentration'], recs['pred_size'], s=230,
                              c=recs['surfactant_HLB'], cmap='RdYlBu',
                              edgecolors='#2a4a6a', lw=1, alpha=0.9)
            ax4.axhline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.7)
            for i in range(len(recs)):
                ax4.annotate(f'F{i+1}', (recs.iloc[i]['surfactant_concentration'], recs.iloc[i]['pred_size']),
                             textcoords='offset points', xytext=(6,4), fontsize=8, color='#8ab0d0')
            ax4.set_xlabel('Surfactant Concentration (%)', fontsize=9)
            ax4.set_ylabel('Predicted Size (nm)', fontsize=9)
            ax4.set_title('Surfactant Conc. vs. Particle Size', fontsize=10, fontweight='bold', pad=12)
            ax4.grid(lw=0.5)
            fig4.colorbar(sc4, ax=ax4, pad=0.01, fraction=0.025).set_label('Surf. HLB', fontsize=8)
            plt.tight_layout(); st.pyplot(fig4)

    # ── Tab 3: Search Space ──
    with tab3:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-green"></span>Search Space Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="abox abox-info">
            All 20,000 candidates. <strong>{hit:,}</strong> ({hit/200:.1f}%) achieved size ≤ {target_size} nm.
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig5, ax5 = plt.subplots(figsize=(7, 4))
            ax5.hist(cdf['pred_size'], bins=60, color='#00d28c', alpha=0.72, edgecolor='none')
            ax5.axvline(target_size, color='#ff6b6b', lw=2, ls='--',
                        label=f'Target: {target_size} nm')
            ax5.axvline(cdf['pred_size'].median(), color='#ffd700', lw=1.5, ls=':',
                        label=f"Median: {cdf['pred_size'].median():.0f} nm")
            ax5.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax5.set_ylabel('Frequency', fontsize=9)
            ax5.set_title('Distribution of Predicted Sizes', fontsize=10, fontweight='bold', pad=12)
            ax5.legend(fontsize=8); ax5.grid(axis='y', lw=0.5)
            plt.tight_layout(); st.pyplot(fig5)
        with c2:
            fig6, ax6 = plt.subplots(figsize=(7, 4))
            ax6.hist(cdf['pred_EE'], bins=60, color='#00aaff', alpha=0.72, edgecolor='none')
            ax6.axvline(min_ee, color='#ff6b6b', lw=2, ls='--', label=f'Min EE: {min_ee}%')
            ax6.axvline(cdf['pred_EE'].median(), color='#ffd700', lw=1.5, ls=':',
                        label=f"Median: {cdf['pred_EE'].median():.1f}%")
            ax6.set_xlabel('Predicted EE%', fontsize=9)
            ax6.set_ylabel('Frequency', fontsize=9)
            ax6.set_title('Distribution of Predicted EE%', fontsize=10, fontweight='bold', pad=12)
            ax6.legend(fontsize=8); ax6.grid(axis='y', lw=0.5)
            plt.tight_layout(); st.pyplot(fig6)

        fig7, ax7 = plt.subplots(figsize=(13, 4.5))
        samp = cdf.sample(3000, random_state=42)
        hit_m = samp['pred_size'] <= target_size
        ax7.scatter(samp[~hit_m]['pred_size'], samp[~hit_m]['pred_EE'],
                    s=5, color='#1e3a5a', alpha=0.3, rasterized=True, label='Outside target')
        ax7.scatter(samp[hit_m]['pred_size'], samp[hit_m]['pred_EE'],
                    s=7, color='#00d28c', alpha=0.45, rasterized=True, label=f'≤ {target_size} nm')
        ax7.scatter(recs['pred_size'], recs['pred_EE'],
                    s=180, color='#ff6b6b', edgecolors='white', lw=1.5, zorder=5,
                    label='Selected recommendations')
        for i in range(len(recs)):
            ax7.annotate(f'F{i+1}', (recs.iloc[i]['pred_size'], recs.iloc[i]['pred_EE']),
                         textcoords='offset points', xytext=(5,5),
                         fontsize=8, fontweight='bold', color='#ff9a9a')
        ax7.axvline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.55)
        ax7.axhline(min_ee,      color='#4a9eff', lw=1.2, ls='--', alpha=0.55)
        ax7.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
        ax7.set_ylabel('Predicted EE%', fontsize=9)
        ax7.set_title('Full Search Space — 3,000 Sample Points (of 20,000 Screened)',
                      fontsize=10, fontweight='bold', pad=12)
        ax7.legend(fontsize=8, loc='upper right'); ax7.grid(lw=0.4)
        plt.tight_layout(); st.pyplot(fig7)

    # ── Tab 4: Protocol ──
    with tab4:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-green"></span>Laboratory Protocol</div>',
                    unsafe_allow_html=True)
        best = recs.iloc[0]
        st.markdown(f"""<div class="abox abox-info">
            Protocol for <strong>F1</strong> — top candidate.
            Predicted size: <strong>{best['pred_size']:.1f} nm</strong>,
            EE%: <strong>{best['pred_EE']:.1f}%</strong>.
            Method: <em>Nanoprecipitation (solvent displacement)</em>.
        </div>""", unsafe_allow_html=True)

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            st.markdown(f"""<div class="proto">
                <div class="proto-head">🔬 Nanoprecipitation — F1 ({best['pred_size']:.1f} nm | EE {best['pred_EE']:.1f}%)</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.66rem;color:rgba(0,210,140,0.42);letter-spacing:2px;margin-bottom:11px">MATERIALS</div>
                <div class="pstep"><div class="pnum">M</div><div class="ptxt">
                  PLGA {best['polymer_MW']:.0f} kDa (LA/GA {best['LA/GA']:.2f}) &nbsp;·&nbsp;
                  Acetonitrile or Acetone &nbsp;·&nbsp; Surfactant (HLB {best['surfactant_HLB']:.1f})
                  &nbsp;·&nbsp; Deionized water &nbsp;·&nbsp; Drug compound
                </div></div>
                <div style="font-family:'Space Mono',monospace;font-size:0.66rem;color:rgba(0,210,140,0.42);letter-spacing:2px;margin:12px 0 8px">PROCEDURE</div>
                <div class="pstep"><div class="pnum">1</div><div class="ptxt">
                  <strong>Organic phase:</strong> Dissolve PLGA {best['polymer_MW']:.0f} kDa in organic solvent.
                  Add drug at drug/polymer ratio = <strong>{best['drug/polymer']:.4f}</strong>.
                </div></div>
                <div class="pstep"><div class="pnum">2</div><div class="ptxt">
                  <strong>Aqueous phase:</strong> Prepare surfactant at <strong>{best['surfactant_concentration']:.3f}%</strong> (w/v).
                  Adjust pH to <strong>{best['pH']:.2f}</strong>.
                </div></div>
                <div class="pstep"><div class="pnum">3</div><div class="ptxt">
                  <strong>Nanoprecipitation:</strong> Add organic phase dropwise to aqueous phase
                  (Aq/Org = <strong>{best['aqueous/organic']:.2f}</strong>),
                  600–800 RPM magnetic stirring, room temperature.
                </div></div>
                <div class="pstep"><div class="pnum">4</div><div class="ptxt">
                  <strong>Solvent removal:</strong> Stir 3–4 h at RT or rotary evaporate at 30°C.
                </div></div>
                <div class="pstep"><div class="pnum">5</div><div class="ptxt">
                  <strong>Purification:</strong> Ultracentrifuge 15,000 × g, 30 min, 4°C. Wash ×3 with DI water.
                </div></div>
                <div class="pstep"><div class="pnum">6</div><div class="ptxt">
                  <strong>Lyophilization:</strong> Freeze-dry with 5% trehalose at −80°C, 48 h. Store at −20°C.
                </div></div>
                <div class="pstep"><div class="pnum">7</div><div class="ptxt">
                  <strong>Characterization:</strong> DLS — size &lt; {target_size} nm, PDI &lt; 0.25,
                  Zeta &gt; |±20 mV|. EE% by UV-Vis post membrane separation. Morphology by TEM.
                </div></div>
            </div>""", unsafe_allow_html=True)

        with col_p2:
            st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;color:#e8f4ff;margin-bottom:10px">All Formulations</div>',
                        unsafe_allow_html=True)
            for i, row in recs.iterrows():
                bc = "#00d28c" if i == 0 else "rgba(255,255,255,0.07)"
                st.markdown(f"""<div class="fcard" style="border-color:{bc}">
                    <div class="fcard-top">
                        <span class="fcard-rank">F{i+1}</span>
                        <span class="fcard-scores">{row['pred_size']:.1f} nm · EE {row['pred_EE']:.1f}%</span>
                    </div>
                    <div class="fcard-params">
                        PLGA {row['polymer_MW']:.0f} kDa · LA/GA {row['LA/GA']:.2f} · D/P {row['drug/polymer']:.4f}<br>
                        Surf. {row['surfactant_concentration']:.3f}% (HLB {row['surfactant_HLB']:.1f}) · Aq/Org {row['aqueous/organic']:.2f} · pH {row['pH']:.2f}
                    </div></div>""", unsafe_allow_html=True)

            proto_out = recs[['drug/polymer','polymer_MW','LA/GA','surfactant_concentration',
                               'surfactant_HLB','aqueous/organic','pH','pred_size','pred_EE']].copy()
            proto_out.insert(0,'Formulation',[f'F{i+1}' for i in range(len(recs))])
            proto_out.columns = ['Formulation','Drug/Polymer','PLGA MW (kDa)','LA/GA',
                                  'Surf. Conc (%)','Surf. HLB','Aq/Org','pH',
                                  'Pred. Size (nm)','Pred. EE (%)']
            st.download_button("📥  Download Protocol CSV", proto_out.to_csv(index=False),
                f"plga_protocol_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# ──────────────────  CHITOSAN OPTIMIZATION  ───────────────────────
# ══════════════════════════════════════════════════════════════════
if run and polymer == 'Chitosan' and models.get('cs_loaded'):

    with st.spinner("Screening 10,000 Chitosan-TPP formulation candidates…"):
        mw_opts   = [chitosan_mw] if fix_mw else [5, 20, 50, 310]
        c_opts    = np.linspace(0.10, 1.0, 25).tolist()
        tpp_opts  = np.linspace(0.10, 1.0, 25).tolist()

        random.seed(42); np.random.seed(42)
        cands = []
        for _ in range(10000):
            mw   = random.choice(mw_opts)
            conc = random.choice(c_opts)
            tpp  = random.choice(tpp_opts)
            cands.append({
                'chitosan_MW': mw, 'chitosan_conc': conc, 'TPP_conc': tpp,
                'chitosan_TPP_ratio': conc/tpp, 'conc_x_TPP': conc*tpp,
                'MW_x_conc': mw*conc, 'MW_x_TPP': mw*tpp,
                'log_MW': np.log10(mw), 'total_solute': conc+tpp,
                'chitosan_fraction': conc/(conc+tpp)
            })

        cdf_cs = pd.DataFrame(cands)
        cdf_cs['pred_size'] = models['cs_size'].predict(cdf_cs[models['cs_features']])
        cdf_cs['score'] = np.where(cdf_cs['pred_size'] <= target_size_cs,
                                   (target_size_cs - cdf_cs['pred_size']) / target_size_cs, -0.5)

        top = cdf_cs.nlargest(n_recs_cs * 4, 'score')
        diverse, used_mw = [], set()
        for _, row in top.iterrows():
            if row['chitosan_MW'] not in used_mw or len(diverse) < 2:
                diverse.append(row); used_mw.add(row['chitosan_MW'])
            if len(diverse) >= n_recs_cs: break
        if len(diverse) < n_recs_cs:
            diverse = [r for _, r in top.head(n_recs_cs).iterrows()]
        recs_cs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)

    hit_cs = (cdf_cs['pred_size'] < target_size_cs).sum()
    MW_MAP = {5:'5 kDa', 20:'20 kDa', 50:'LMW (~50 kDa)', 310:'HMW (~310 kDa)'}

    st.markdown('<div class="abox abox-success">✅ &nbsp; Chitosan-TPP optimization complete — 10,000 candidates screened.</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi"><div class="kpi-val">10,000</div><div class="kpi-lbl">Screened</div></div>
      <div class="kpi"><div class="kpi-val">{hit_cs:,}</div><div class="kpi-lbl">Within Target</div></div>
      <div class="kpi"><div class="kpi-val">{recs_cs['pred_size'].min():.1f}<span class="kpi-unit">nm</span></div><div class="kpi-lbl">Best Size</div></div>
      <div class="kpi"><div class="kpi-val">{n_recs_cs}</div><div class="kpi-lbl">Recommendations</div></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📋  Recommendations", "📊  Visual Analysis", "🧪  Lab Protocol"
    ])

    # ── Tab 1 ──
    with tab1:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-blue"></span>Top Recommended Chitosan Formulations</div>',
                    unsafe_allow_html=True)
        disp_cs = recs_cs[['chitosan_MW','chitosan_conc','TPP_conc',
                             'chitosan_TPP_ratio','pred_size']].copy()
        disp_cs.insert(0, 'Rank', [f'F{i+1}' for i in range(len(disp_cs))])
        disp_cs.insert(1, 'CS Type', recs_cs['chitosan_MW'].map(MW_MAP))
        disp_cs.columns = ['Rank','CS Type','MW (kDa)','CS Conc (mg/mL)',
                            'TPP Conc (mg/mL)','CS:TPP Ratio','Pred. Size (nm)']
        st.dataframe(
            disp_cs.style
                .format({'MW (kDa)':'{:.0f}','CS Conc (mg/mL)':'{:.2f}',
                         'TPP Conc (mg/mL)':'{:.2f}','CS:TPP Ratio':'{:.2f}','Pred. Size (nm)':'{:.1f}'})
                .background_gradient(subset=['Pred. Size (nm)'], cmap='RdYlGn_r', vmin=50, vmax=250),
            use_container_width=True, height=min(360, 60+len(disp_cs)*52)
        )
        st.download_button("📥  Download CSV", disp_cs.to_csv(index=False),
            f"chitosan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    # ── Tab 2 ──
    with tab2:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-blue"></span>Visual Analysis</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            bar_colors = ['#00d28c' if s < target_size_cs else '#ff6b6b' for s in recs_cs['pred_size']]
            ax.barh(range(len(recs_cs)), recs_cs['pred_size'],
                    color=bar_colors, height=0.62, edgecolor='none')
            ax.axvline(target_size_cs, color='#ff6b6b', lw=1.5, ls='--', alpha=0.85,
                       label=f'Target: {target_size_cs} nm')
            ax.set_yticks(range(len(recs_cs)))
            ax.set_yticklabels([f'F{i+1}' for i in range(len(recs_cs))], fontsize=9)
            ax.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax.set_title('Chitosan-TPP — Ranked by Size', fontsize=10, fontweight='bold', pad=12)
            p1 = mpatches.Patch(color='#00d28c', label=f'≤ {target_size_cs} nm ✓')
            p2 = mpatches.Patch(color='#ff6b6b', label=f'> {target_size_cs} nm')
            ax.legend(handles=[p1,p2], fontsize=8)
            ax.grid(axis='x', lw=0.5)
            for i, v in enumerate(recs_cs['pred_size']):
                ax.text(v+1.5, i, f'{v:.1f}', va='center', fontsize=8, color='#8ab8d8')
            plt.tight_layout(); st.pyplot(fig)

        with c2:
            MW_COL = {5:'#f0b429', 20:'#4a9eff', 50:'#00d28c', 310:'#ff6b6b'}
            fig2, ax2 = plt.subplots(figsize=(7, 4.8))
            for mw_val in recs_cs['chitosan_MW'].unique():
                mask = recs_cs['chitosan_MW'] == mw_val
                ax2.scatter(recs_cs[mask]['chitosan_conc'], recs_cs[mask]['pred_size'],
                            s=220, color=MW_COL.get(mw_val,'gray'),
                            edgecolors='white', lw=1, alpha=0.9,
                            label=MW_MAP.get(mw_val,str(mw_val)))
            ax2.axhline(target_size_cs, color='#ff6b6b', lw=1.5, ls='--', alpha=0.7,
                        label=f'Target: {target_size_cs} nm')
            ax2.set_xlabel('Chitosan Concentration (mg/mL)', fontsize=9)
            ax2.set_ylabel('Predicted Size (nm)', fontsize=9)
            ax2.set_title('CS Concentration vs. Size by Grade', fontsize=10, fontweight='bold', pad=12)
            ax2.legend(fontsize=8); ax2.grid(lw=0.5)
            plt.tight_layout(); st.pyplot(fig2)

        # Distribution
        fig3, ax3 = plt.subplots(figsize=(13, 3.8))
        ax3.hist(cdf_cs['pred_size'], bins=55, color='#0096ff', alpha=0.7, edgecolor='none')
        ax3.axvline(target_size_cs, color='#ff6b6b', lw=2, ls='--',
                    label=f'Target: {target_size_cs} nm')
        ax3.axvline(cdf_cs['pred_size'].median(), color='#ffd700', lw=1.5, ls=':',
                    label=f"Median: {cdf_cs['pred_size'].median():.0f} nm")
        ax3.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
        ax3.set_ylabel('Frequency', fontsize=9)
        ax3.set_title('Distribution of All 10,000 Predicted Sizes', fontsize=10, fontweight='bold', pad=12)
        ax3.legend(fontsize=8); ax3.grid(axis='y', lw=0.5)
        plt.tight_layout(); st.pyplot(fig3)

    # ── Tab 3 ──
    with tab3:
        st.markdown('<div class="sec-title"><span class="sec-dot dot-blue"></span>Laboratory Protocol</div>',
                    unsafe_allow_html=True)
        best_cs = recs_cs.iloc[0]
        st.markdown(f"""<div class="abox abox-info">
            Protocol for <strong>F1</strong> — top Chitosan-TPP candidate.
            Predicted size: <strong>{best_cs['pred_size']:.1f} nm</strong>.
            Method: <em>Ionic Gelation</em>.
        </div>""", unsafe_allow_html=True)

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            st.markdown(f"""<div class="proto">
                <div class="proto-head">🔬 Ionic Gelation Protocol — F1 ({best_cs['pred_size']:.1f} nm)</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.66rem;color:rgba(0,150,255,0.42);letter-spacing:2px;margin-bottom:11px">MATERIALS</div>
                <div class="pstep"><div class="pnum">M</div><div class="ptxt">
                  {MW_MAP.get(best_cs['chitosan_MW'],'')} Chitosan &nbsp;·&nbsp; Sodium Tripolyphosphate (TPP)
                  &nbsp;·&nbsp; 1% Glacial acetic acid &nbsp;·&nbsp; Deionized water
                </div></div>
                <div style="font-family:'Space Mono',monospace;font-size:0.66rem;color:rgba(0,150,255,0.42);letter-spacing:2px;margin:12px 0 8px">PROCEDURE</div>
                <div class="pstep"><div class="pnum">1</div><div class="ptxt">
                  <strong>Chitosan solution:</strong> Dissolve {MW_MAP.get(best_cs['chitosan_MW'],'')} chitosan
                  at <strong>{best_cs['chitosan_conc']:.2f} mg/mL</strong> in 1% glacial acetic acid.
                  Stir overnight for complete dissolution.
                </div></div>
                <div class="pstep"><div class="pnum">2</div><div class="ptxt">
                  <strong>TPP solution:</strong> Dissolve sodium TPP at
                  <strong>{best_cs['TPP_conc']:.2f} mg/mL</strong> in deionized water.
                  (CS:TPP = <strong>{best_cs['chitosan_TPP_ratio']:.2f}</strong>)
                </div></div>
                <div class="pstep"><div class="pnum">3</div><div class="ptxt">
                  <strong>Ionic gelation:</strong> Add TPP solution dropwise into chitosan solution
                  under magnetic stirring at 500–700 RPM, room temperature.
                  Nanoparticles form spontaneously via electrostatic interaction.
                </div></div>
                <div class="pstep"><div class="pnum">4</div><div class="ptxt">
                  <strong>Equilibration:</strong> Continue stirring for 30 min at room temperature.
                  No heating required.
                </div></div>
                <div class="pstep"><div class="pnum">5</div><div class="ptxt">
                  <strong>Characterization:</strong> DLS — size &lt; {target_size_cs} nm,
                  PDI &lt; 0.30, Zeta potential &gt; +20 mV (cationic stability indicator).
                </div></div>
            </div>""", unsafe_allow_html=True)

        with col_p2:
            st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;color:#e8f4ff;margin-bottom:10px">All Formulations</div>',
                        unsafe_allow_html=True)
            for i, row in recs_cs.iterrows():
                bc = "#0096ff" if i == 0 else "rgba(255,255,255,0.07)"
                st.markdown(f"""<div class="fcard" style="border-color:{bc}">
                    <div class="fcard-top">
                        <span class="fcard-rank">F{i+1}</span>
                        <span class="fcard-scores" style="color:#0096ff">{row['pred_size']:.1f} nm</span>
                    </div>
                    <div class="fcard-params">
                        {MW_MAP.get(row['chitosan_MW'],'')} · CS {row['chitosan_conc']:.2f} mg/mL
                        · TPP {row['TPP_conc']:.2f} mg/mL<br>
                        CS:TPP {row['chitosan_TPP_ratio']:.2f}
                    </div></div>""", unsafe_allow_html=True)

            proto_cs = recs_cs[['chitosan_MW','chitosan_conc','TPP_conc',
                                  'chitosan_TPP_ratio','pred_size']].copy()
            proto_cs.insert(0,'Formulation',[f'F{i+1}' for i in range(len(recs_cs))])
            proto_cs.insert(1,'CS Type', recs_cs['chitosan_MW'].map(MW_MAP))
            proto_cs.columns = ['Formulation','CS Type','MW (kDa)','CS Conc (mg/mL)',
                                  'TPP Conc (mg/mL)','CS:TPP Ratio','Pred. Size (nm)']
            st.download_button("📥  Download Protocol CSV", proto_cs.to_csv(index=False),
                f"chitosan_protocol_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  NANOFORMULA AI &nbsp;·&nbsp; PLGA &amp; CHITOSAN-TPP OPTIMIZER &nbsp;·&nbsp;
  PLGA R²=0.88 · CS R²=0.83 &nbsp;·&nbsp;
  DEVELOPED BY HARDIK SOOD &nbsp;·&nbsp; IIT (BHU) VARANASI
</div>
""", unsafe_allow_html=True)

