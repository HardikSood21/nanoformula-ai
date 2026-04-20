      """
NanoFormula AI — PLGA Nanoparticle Formulation Optimizer
ML-driven virtual screening for particle size and encapsulation efficiency
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
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NanoFormula AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: #03070f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,200,150,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,120,255,0.06) 0%, transparent 60%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0; padding-bottom: 4rem; max-width: 1280px; }

.hero { padding: 3.5rem 2rem 2rem; text-align: center; }
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase;
    color: rgba(0,210,140,0.7); margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: center; gap: 10px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: ''; width: 40px; height: 1px; background: rgba(0,210,140,0.3);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem); font-weight: 800;
    line-height: 0.95; letter-spacing: -3px;
    background: linear-gradient(160deg, #ffffff 20%, #00d28c 60%, #00aaff 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1.2rem;
}
.hero-desc {
    font-size: 1.05rem; font-weight: 300;
    color: rgba(180,210,240,0.55); max-width: 560px;
    margin: 0 auto 2.5rem; line-height: 1.75; font-style: italic;
}
.hero-stats {
    display: flex; justify-content: center; gap: 0; flex-wrap: wrap;
    border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
    overflow: hidden; max-width: 720px; margin: 0 auto 1rem;
    backdrop-filter: blur(10px); background: rgba(255,255,255,0.025);
}
.hero-stat {
    flex: 1; min-width: 120px; padding: 1.2rem 1rem; text-align: center;
    border-right: 1px solid rgba(255,255,255,0.06);
}
.hero-stat:last-child { border-right: none; }
.hero-stat-val {
    font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700;
    color: #00d28c; line-height: 1; margin-bottom: 4px;
}
.hero-stat-lbl {
    font-size: 0.68rem; color: rgba(150,185,220,0.4);
    letter-spacing: 1.8px; text-transform: uppercase; font-family: 'Space Mono', monospace;
}

.sec-title {
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: rgba(200,230,255,0.85); letter-spacing: -0.3px;
    padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0 1.2rem; display: flex; align-items: center; gap: 8px;
}
.sec-dot { width: 6px; height: 6px; border-radius: 50%; background: #00d28c; flex-shrink: 0; }

.glass-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
}
.result-strip {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.2rem 0;
}
.r-card {
    background: rgba(0,210,140,0.05); border: 1px solid rgba(0,210,140,0.15);
    border-radius: 14px; padding: 1.2rem 1rem; text-align: center;
}
.r-val {
    font-family: 'Space Mono', monospace; font-size: 1.55rem; font-weight: 700;
    color: #00d28c; line-height: 1;
}
.r-unit { font-size: 0.75rem; color: rgba(0,210,140,0.5); margin-left: 2px; }
.r-lbl {
    font-size: 0.65rem; color: rgba(150,185,220,0.4);
    letter-spacing: 1.5px; text-transform: uppercase;
    font-family: 'Space Mono', monospace; margin-top: 6px;
}

.box-info {
    background: rgba(0,150,255,0.07); border-left: 3px solid #0096ff;
    border-radius: 0 10px 10px 0; padding: 11px 15px;
    font-size: 0.875rem; color: rgba(180,220,255,0.8); line-height: 1.6; margin: 10px 0;
}
.box-success {
    background: rgba(0,210,140,0.07); border-left: 3px solid #00d28c;
    border-radius: 0 10px 10px 0; padding: 11px 15px;
    font-size: 0.875rem; color: rgba(160,255,210,0.85); line-height: 1.6; margin: 10px 0;
}

.protocol {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem 1.8rem;
}
.protocol-head {
    font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700;
    color: #00d28c; margin-bottom: 1.2rem;
}
.p-step { display: flex; gap: 14px; margin-bottom: 12px; align-items: flex-start; }
.p-num {
    width: 26px; height: 26px; background: rgba(0,210,140,0.12);
    border: 1px solid rgba(0,210,140,0.25); color: #00d28c; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; flex-shrink: 0;
    font-family: 'Space Mono', monospace;
}
.p-text { font-size: 0.875rem; color: rgba(190,220,245,0.75); line-height: 1.65; }

.feat-row {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feat-name { font-size: 0.8rem; color: rgba(180,210,240,0.65); min-width: 200px; }
.feat-bar-bg { flex: 1; height: 5px; background: rgba(255,255,255,0.06); border-radius: 3px; }
.feat-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #00d28c, #00aaff); }
.feat-pct { font-size: 0.75rem; color: rgba(0,210,140,0.7); font-family: 'Space Mono', monospace; min-width: 38px; text-align: right; }

[data-testid="stSidebar"] {
    background: rgba(3, 8, 18, 0.98) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color: rgba(170,200,230,0.65) !important; font-size: 0.83rem !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: rgba(210,235,255,0.85) !important; font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    letter-spacing: 0.5px; text-transform: uppercase;
}

.stButton > button {
    background: linear-gradient(135deg, #00d28c, #00aaff) !important;
    color: #03070f !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    letter-spacing: 1px !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.4rem !important;
    transition: all 0.25s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,210,140,0.3) !important; }
.stDownloadButton > button {
    background: rgba(0,210,140,0.08) !important; color: #00d28c !important;
    border: 1px solid rgba(0,210,140,0.3) !important; border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: #00d28c !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: rgba(150,185,215,0.45) !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; padding: 4px !important; gap: 2px !important; }
.stTabs [data-baseweb="tab"] { color: rgba(160,195,230,0.5) !important; font-weight: 500 !important; border-radius: 9px !important; font-size: 0.85rem !important; }
.stTabs [aria-selected="true"] { background: rgba(0,210,140,0.12) !important; color: #00d28c !important; }
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }
.stSpinner > div { border-top-color: #00d28c !important; }

.app-footer {
    border-top: 1px solid rgba(255,255,255,0.05); margin-top: 4rem;
    padding: 1.5rem 0 0.5rem; text-align: center;
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 1px; color: rgba(120,160,200,0.25);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#080f1c', 'axes.facecolor': '#0a1220',
    'axes.edgecolor': '#182840', 'axes.labelcolor': '#7aa8cc',
    'xtick.color': '#4a6a8a', 'ytick.color': '#4a6a8a',
    'text.color': '#b0d0f0', 'grid.color': '#182840', 'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    m = {}
    try:
        with open('model_particle_size_final.pkl', 'rb') as f: m['size'] = pickle.load(f)
        with open('model_ee_final.pkl', 'rb') as f:           m['ee']   = pickle.load(f)
        m['data']   = pd.read_csv('PLGA_nanoparticles_dataset.csv')
        m['loaded'] = True
    except Exception as e:
        m['loaded'] = False; m['error'] = str(e)
    return m

models = load_models()
if not models['loaded']:
    st.error(f"⚠️ Could not load model files. Error: {models.get('error','Unknown')}")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Machine Learning · Drug Delivery · Nanomedicine</div>
    <div class="hero-title">NanoFormula<br>AI</div>
    <div class="hero-desc">
        Virtual screening of 20,000 PLGA nanoparticle formulations.<br>
        Predict particle size &amp; encapsulation efficiency in seconds.
    </div>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-val">433</div><div class="hero-stat-lbl">Training Sets</div></div>
        <div class="hero-stat"><div class="hero-stat-val">0.88</div><div class="hero-stat-lbl">Size R²</div></div>
        <div class="hero-stat"><div class="hero-stat-val">±22nm</div><div class="hero-stat-lbl">Size MAE</div></div>
        <div class="hero-stat"><div class="hero-stat-val">20K</div><div class="hero-stat-lbl">Candidates</div></div>
        <div class="hero-stat"><div class="hero-stat-val">65</div><div class="hero-stat-lbl">Drugs Trained</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="padding:10px 0 6px">
  <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#e8f4ff;letter-spacing:-0.5px">🧬 NanoFormula AI</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:rgba(0,210,140,0.5);letter-spacing:2px;margin-top:2px">PLGA OPTIMIZER · v3.0</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("### Drug Properties")
c1, c2 = st.sidebar.columns(2)
with c1:
    mol_MW   = st.number_input("MW (g/mol)",     100.0, 1000.0, 420.5, 10.0)
    mol_logP = st.number_input("LogP",           -5.0,  10.0,   3.1,   0.1)
    mol_TPSA = st.number_input("TPSA (Ų)",       0.0,   300.0,  72.0,  5.0)
    mol_mp   = st.number_input("Melting Pt (°C)", 0.0,  500.0,  175.0, 5.0)
with c2:
    mol_Hacc = st.number_input("H-Acceptors",    0, 20, 5, 1)
    mol_Hdon = st.number_input("H-Donors",       0, 10, 2, 1)
    mol_het  = st.number_input("Heteroatoms",    0, 30, 7, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### PLGA Grade")
polymer_mode = st.sidebar.radio("Selection:", ["Auto-optimize", "Specify grade"])
polymer_constraints = None
if polymer_mode == "Specify grade":
    df_ = models['data']
    sel_mw   = st.sidebar.selectbox("PLGA MW (kDa)", sorted(df_['polymer_MW'].unique()))
    sel_laga = st.sidebar.selectbox("LA/GA Ratio",   sorted(df_['LA/GA'].unique()))
    polymer_constraints = {'polymer_MW': sel_mw, 'LA/GA': sel_laga}

st.sidebar.markdown("---")
st.sidebar.markdown("### Optimization Targets")
target_size = st.sidebar.slider("Target Particle Size (nm)", 50, 300, 180, 5)
min_ee      = st.sidebar.slider("Minimum EE%",               40, 95,  70,  5)
n_recs      = st.sidebar.slider("No. of Recommendations",    3,  10,  5,   1)
size_weight = st.sidebar.slider("Size ← Priority → EE",      0.3, 0.9, 0.7, 0.05,
                                 help="0.7 = 70% weight on particle size, 30% on EE%")
st.sidebar.markdown("---")
run = st.sidebar.button("▶  Run Optimization", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PRE-RUN VIEW
# ══════════════════════════════════════════════════════════════════
if not run:
    st.markdown('<div class="sec-title"><span class="sec-dot"></span>Model Architecture</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="glass-card">
            <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#e8f4ff;margin-bottom:10px">Particle Size Model</div>
            <div class="feat-row"><span class="feat-name">Algorithm</span><span style="color:#00d28c;font-size:0.8rem">Random Forest</span></div>
            <div class="feat-row"><span class="feat-name">R² Score</span><span style="color:#00d28c;font-size:0.8rem">0.88</span></div>
            <div class="feat-row"><span class="feat-name">MAE</span><span style="color:#00d28c;font-size:0.8rem">±22 nm</span></div>
            <div class="feat-row"><span class="feat-name">Training Samples</span><span style="color:#00d28c;font-size:0.8rem">433</span></div>
            <div class="feat-row"><span class="feat-name">Input Features</span><span style="color:#00d28c;font-size:0.8rem">15</span></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="glass-card">
            <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#e8f4ff;margin-bottom:10px">EE% Model</div>
            <div class="feat-row"><span class="feat-name">Algorithm</span><span style="color:#00aaff;font-size:0.8rem">Random Forest</span></div>
            <div class="feat-row"><span class="feat-name">R² Score</span><span style="color:#00aaff;font-size:0.8rem">0.47</span></div>
            <div class="feat-row"><span class="feat-name">Training Samples</span><span style="color:#00aaff;font-size:0.8rem">433</span></div>
            <div class="feat-row"><span class="feat-name">Drug Classes</span><span style="color:#00aaff;font-size:0.8rem">65</span></div>
            <div class="feat-row"><span class="feat-name">Output Range</span><span style="color:#00aaff;font-size:0.8rem">0–100%</span></div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="glass-card">
            <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#e8f4ff;margin-bottom:10px">Virtual Screening</div>
            <div class="feat-row"><span class="feat-name">Candidates / Run</span><span style="color:#a78bfa;font-size:0.8rem">20,000</span></div>
            <div class="feat-row"><span class="feat-name">Sampling Method</span><span style="color:#a78bfa;font-size:0.8rem">Monte Carlo</span></div>
            <div class="feat-row"><span class="feat-name">Diversity Filter</span><span style="color:#a78bfa;font-size:0.8rem">Yes</span></div>
            <div class="feat-row"><span class="feat-name">Scoring</span><span style="color:#a78bfa;font-size:0.8rem">Weighted Multi-obj.</span></div>
            <div class="feat-row"><span class="feat-name">Output</span><span style="color:#a78bfa;font-size:0.8rem">Top-N Formulations</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title"><span class="sec-dot"></span>Input Feature Importance (Relative)</div>', unsafe_allow_html=True)
    features_info = [
        ("Polymer MW (kDa)", 92), ("LA/GA Ratio", 88), ("Drug/Polymer Ratio", 85),
        ("Surfactant Concentration", 80), ("Surfactant HLB", 72), ("Drug LogP", 68),
        ("Aqueous/Organic Ratio", 65), ("Drug Molecular Weight", 60),
        ("pH", 54), ("Solvent Polarity Index", 48),
    ]
    feat_html = "".join([
        f'<div class="feat-row"><span class="feat-name">{n}</span>'
        f'<div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{p}%"></div></div>'
        f'<span class="feat-pct">{p}%</span></div>'
        for n, p in features_info
    ])
    st.markdown(f'<div class="glass-card">{feat_html}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="box-info">
        <strong>Getting started:</strong> Enter your drug's molecular properties in the sidebar, set your target
        particle size and minimum EE%, then click <strong>▶ Run Optimization</strong>.
        The model screens 20,000 candidate PLGA formulations via Monte Carlo sampling and surfaces the top
        recommendations using a weighted multi-objective score.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ══════════════════════════════════════════════════════════════════
if run:
    drug_props = {
        'mol_MW': mol_MW, 'mol_logP': mol_logP, 'mol_TPSA': mol_TPSA,
        'mol_melting_point': mol_mp, 'mol_Hacceptors': mol_Hacc,
        'mol_Hdonors': mol_Hdon, 'mol_heteroatoms': mol_het
    }

    with st.spinner("Screening 20,000 PLGA formulation candidates…"):
        df_plga    = models['data']
        successful = df_plga[df_plga['particle_size'] < target_size]
        if len(successful) < 20:
            successful = df_plga[df_plga['particle_size'] < 300]

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
        candidates = [{
            **drug_props,
            'polymer_MW':             random.choice(space['polymer_MW']),
            'LA/GA':                  random.choice(space['LA/GA']),
            'drug/polymer':           random.choice(space['drug/polymer']),
            'surfactant_concentration': random.choice(space['surfactant_concentration']),

