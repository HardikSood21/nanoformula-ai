"""
NanoFormula AI — PLGA Nanoparticle Formulation Optimizer
ML-driven virtual screening for particle size and entrapment efficiency
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
            'surfactant_HLB':         random.choice(space['surfactant_HLB']),
            'aqueous/organic':        random.choice(space['aqueous/organic']),
            'pH':                     random.choice(space['pH']),
            'solvent_polarity_index': random.choice(space['solvent_polarity_index']),
        } for _ in range(20000)]

        FEATURES = ['polymer_MW','LA/GA','mol_MW','mol_logP','mol_TPSA',
                    'mol_melting_point','mol_Hacceptors','mol_Hdonors','mol_heteroatoms',
                    'drug/polymer','surfactant_concentration','surfactant_HLB',
                    'aqueous/organic','pH','solvent_polarity_index']
        cdf = pd.DataFrame(candidates)
        cdf['pred_size'] = models['size'].predict(cdf[FEATURES])
        cdf['pred_EE']   = np.clip(models['ee'].predict(cdf[FEATURES]), 0, 100)

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

    hit_rate = (cdf['pred_size'] < target_size).sum()

    st.markdown('<div class="box-success">✅ &nbsp; Optimization complete — 20,000 candidates screened. Top formulations ranked below.</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-strip">
        <div class="r-card"><div class="r-val">20,000</div><div class="r-lbl">Candidates</div></div>
        <div class="r-card"><div class="r-val">{hit_rate:,}</div><div class="r-lbl">Within Target</div></div>
        <div class="r-card"><div class="r-val">{recs['pred_size'].min():.1f}<span class="r-unit">nm</span></div><div class="r-lbl">Best Size</div></div>
        <div class="r-card"><div class="r-val">{recs['pred_EE'].max():.1f}<span class="r-unit">%</span></div><div class="r-lbl">Best EE%</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋  Recommendations", "📊  Visual Analysis", "🔭  Search Space", "🧪  Lab Protocol"
    ])

    # ─── TAB 1: TABLE ───────────────────────────────────────────
    with tab1:
        st.markdown('<div class="sec-title"><span class="sec-dot"></span>Top Recommended Formulations</div>',
                    unsafe_allow_html=True)
        display = recs[['drug/polymer','polymer_MW','LA/GA','surfactant_concentration',
                         'surfactant_HLB','aqueous/organic','pH','pred_size','pred_EE','score']].copy()
        display.insert(0, 'Rank', [f'F{i+1}' for i in range(len(display))])
        display.columns = ['Rank','Drug/Polymer','PLGA MW (kDa)','LA/GA','Surf. Conc (%)','Surf. HLB',
                            'Aq/Org','pH','Pred. Size (nm)','Pred. EE (%)','Score']
        st.dataframe(
            display.style
                .format({'Drug/Polymer':'{:.4f}','PLGA MW (kDa)':'{:.1f}','LA/GA':'{:.2f}',
                         'Surf. Conc (%)':'{:.3f}','Surf. HLB':'{:.1f}','Aq/Org':'{:.2f}',
                         'pH':'{:.2f}','Pred. Size (nm)':'{:.1f}','Pred. EE (%)':'{:.1f}','Score':'{:.4f}'})
                .background_gradient(subset=['Pred. Size (nm)'], cmap='RdYlGn_r', vmin=50, vmax=300)
                .background_gradient(subset=['Pred. EE (%)'],    cmap='RdYlGn',   vmin=40, vmax=100)
                .background_gradient(subset=['Score'],           cmap='Blues'),
            use_container_width=True, height=320
        )
        st.markdown("""<div class="box-info">
            <strong>Score</strong> = weighted sum of size and EE% sub-scores. Higher = better overall.
            Diversity filter ensures recommendations span varied drug/polymer ratios.
        </div>""", unsafe_allow_html=True)
        st.download_button("📥  Download CSV",
            display.to_csv(index=False),
            f"nanoformula_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    # ─── TAB 2: CHARTS ──────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-title"><span class="sec-dot"></span>Visual Analysis</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            norm   = plt.Normalize(recs['pred_EE'].min(), recs['pred_EE'].max())
            colors = plt.cm.cool(norm(recs['pred_EE']))
            ax.barh(range(len(recs)), recs['pred_size'], color=colors, height=0.6, edgecolor='none')
            ax.axvline(target_size, color='#ff6b6b', lw=1.5, ls='--', alpha=0.85, label=f'Target: {target_size} nm')
            ax.set_yticks(range(len(recs)))
            ax.set_yticklabels([f"F{i+1}" for i in range(len(recs))], fontsize=9)
            ax.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax.set_title('Ranked Formulations by Particle Size', fontsize=10, fontweight='bold', pad=12)
            ax.legend(fontsize=8); ax.grid(axis='x', lw=0.5)
            for i, v in enumerate(recs['pred_size']):
                ax.text(v + 1.5, i, f'{v:.1f}', va='center', fontsize=8, color='#90b8d8')
            sm = cm.ScalarMappable(cmap='cool', norm=norm); sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
            cb.set_label('EE%', fontsize=8); cb.ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(7, 4.8))
            sc = ax2.scatter(recs['pred_size'], recs['pred_EE'], s=350,
                             c=recs['score'], cmap='plasma',
                             edgecolors='#3a6a9a', linewidth=1.2, alpha=0.92, zorder=3)
            ax2.axvspan(0, target_size, alpha=0.04, color='#00d28c')
            ax2.axhspan(min_ee, 100,    alpha=0.04, color='#00aaff')
            ax2.axvline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.6)
            ax2.axhline(min_ee,      color='#4a9eff', lw=1.2, ls='--', alpha=0.6)
            for i in range(len(recs)):
                ax2.annotate(f'F{i+1}', (recs.iloc[i]['pred_size'], recs.iloc[i]['pred_EE']),
                             fontsize=9, fontweight='bold', ha='center', va='center', color='white')
            ax2.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax2.set_ylabel('Predicted EE%', fontsize=9)
            ax2.set_title('Pareto Space: Size vs. Encapsulation Efficiency', fontsize=10, fontweight='bold', pad=12)
            ax2.grid(lw=0.5)
            cb2 = fig2.colorbar(sc, ax=ax2, pad=0.01, fraction=0.025)
            cb2.set_label('Score', fontsize=8); cb2.ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig2)

        c3, c4 = st.columns(2)
        with c3:
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sc3 = ax3.scatter(recs['drug/polymer'], recs['pred_size'], s=250,
                              c=recs['pred_EE'], cmap='YlGn',
                              edgecolors='#2a4a6a', linewidth=1, alpha=0.9)
            ax3.axhline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.7)
            for i in range(len(recs)):
                ax3.annotate(f'F{i+1}', (recs.iloc[i]['drug/polymer'], recs.iloc[i]['pred_size']),
                             textcoords='offset points', xytext=(6, 4), fontsize=8, color='#8ab0d0')
            ax3.set_xlabel('Drug/Polymer Ratio', fontsize=9)
            ax3.set_ylabel('Predicted Particle Size (nm)', fontsize=9)
            ax3.set_title('Drug/Polymer Ratio vs. Particle Size', fontsize=10, fontweight='bold', pad=12)
            ax3.grid(lw=0.5)
            cb3 = fig3.colorbar(sc3, ax=ax3, pad=0.01, fraction=0.025)
            cb3.set_label('EE%', fontsize=8); cb3.ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig3)

        with c4:
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            sc4 = ax4.scatter(recs['surfactant_concentration'], recs['pred_size'], s=250,
                              c=recs['surfactant_HLB'], cmap='RdYlBu',
                              edgecolors='#2a4a6a', linewidth=1, alpha=0.9)
            ax4.axhline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.7)
            for i in range(len(recs)):
                ax4.annotate(f'F{i+1}', (recs.iloc[i]['surfactant_concentration'], recs.iloc[i]['pred_size']),
                             textcoords='offset points', xytext=(6, 4), fontsize=8, color='#8ab0d0')
            ax4.set_xlabel('Surfactant Concentration (%)', fontsize=9)
            ax4.set_ylabel('Predicted Particle Size (nm)', fontsize=9)
            ax4.set_title('Surfactant Concentration vs. Particle Size', fontsize=10, fontweight='bold', pad=12)
            ax4.grid(lw=0.5)
            cb4 = fig4.colorbar(sc4, ax=ax4, pad=0.01, fraction=0.025)
            cb4.set_label('Surfactant HLB', fontsize=8); cb4.ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig4)

    # ─── TAB 3: SEARCH SPACE ────────────────────────────────────
    with tab3:
        st.markdown('<div class="sec-title"><span class="sec-dot"></span>Full Search Space Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="box-info">
            Distribution of all <strong>20,000</strong> screened candidates.
            <strong>{hit_rate:,}</strong> candidates ({hit_rate/200:.1f}%) achieved size ≤ {target_size} nm.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig5, ax5 = plt.subplots(figsize=(7, 4))
            ax5.hist(cdf['pred_size'], bins=60, color='#00d28c', alpha=0.7, edgecolor='none')
            ax5.axvline(target_size, color='#ff6b6b', lw=2, ls='--', label=f'Target: {target_size} nm')
            ax5.axvline(cdf['pred_size'].median(), color='#ffd700', lw=1.5, ls=':',
                        label=f"Median: {cdf['pred_size'].median():.0f} nm")
            ax5.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
            ax5.set_ylabel('Frequency', fontsize=9)
            ax5.set_title('Distribution of Predicted Particle Sizes', fontsize=10, fontweight='bold', pad=12)
            ax5.legend(fontsize=8); ax5.grid(axis='y', lw=0.5)
            plt.tight_layout(); st.pyplot(fig5)

        with c2:
            fig6, ax6 = plt.subplots(figsize=(7, 4))
            ax6.hist(cdf['pred_EE'], bins=60, color='#00aaff', alpha=0.7, edgecolor='none')
            ax6.axvline(min_ee, color='#ff6b6b', lw=2, ls='--', label=f'Min EE: {min_ee}%')
            ax6.axvline(cdf['pred_EE'].median(), color='#ffd700', lw=1.5, ls=':',
                        label=f"Median: {cdf['pred_EE'].median():.1f}%")
            ax6.set_xlabel('Predicted Encapsulation Efficiency (%)', fontsize=9)
            ax6.set_ylabel('Frequency', fontsize=9)
            ax6.set_title('Distribution of Predicted EE%', fontsize=10, fontweight='bold', pad=12)
            ax6.legend(fontsize=8); ax6.grid(axis='y', lw=0.5)
            plt.tight_layout(); st.pyplot(fig6)

        fig7, ax7 = plt.subplots(figsize=(14, 4.5))
        sample   = cdf.sample(3000, random_state=42)
        hit_mask = sample['pred_size'] <= target_size
        ax7.scatter(sample[~hit_mask]['pred_size'], sample[~hit_mask]['pred_EE'],
                    s=5, color='#1e3a5a', alpha=0.3, label='Outside target', rasterized=True)
        ax7.scatter(sample[hit_mask]['pred_size'], sample[hit_mask]['pred_EE'],
                    s=8, color='#00d28c', alpha=0.45, label=f'≤ {target_size} nm', rasterized=True)
        ax7.scatter(recs['pred_size'], recs['pred_EE'],
                    s=200, color='#ff6b6b', edgecolors='white', linewidth=1.5,
                    zorder=5, label='Selected recommendations')
        for i in range(len(recs)):
            ax7.annotate(f'F{i+1}', (recs.iloc[i]['pred_size'], recs.iloc[i]['pred_EE']),
                         textcoords='offset points', xytext=(5, 5),
                         fontsize=8, fontweight='bold', color='#ff9a9a')
        ax7.axvline(target_size, color='#ff6b6b', lw=1.2, ls='--', alpha=0.6)
        ax7.axhline(min_ee,      color='#4a9eff', lw=1.2, ls='--', alpha=0.6)
        ax7.set_xlabel('Predicted Particle Size (nm)', fontsize=9)
        ax7.set_ylabel('Predicted EE%', fontsize=9)
        ax7.set_title('Full Search Space — 3,000 Sample Points (of 20,000 Screened)', fontsize=10, fontweight='bold', pad=12)
        ax7.legend(fontsize=8, loc='upper right'); ax7.grid(lw=0.4)
        plt.tight_layout(); st.pyplot(fig7)

    # ─── TAB 4: PROTOCOL ────────────────────────────────────────
    with tab4:
        st.markdown('<div class="sec-title"><span class="sec-dot"></span>Suggested Laboratory Protocol</div>',
                    unsafe_allow_html=True)
        best = recs.iloc[0]
        st.markdown(f"""<div class="box-info">
            Protocol for <strong>Formulation F1</strong> — top-ranked candidate.
            Predicted particle size: <strong>{best['pred_size']:.1f} nm</strong>,
            EE%: <strong>{best['pred_EE']:.1f}%</strong>.
            Based on <em>nanoprecipitation (solvent displacement)</em> method.
        </div>""", unsafe_allow_html=True)

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            st.markdown(f"""
            <div class="protocol">
                <div class="protocol-head">🔬 Nanoprecipitation Protocol — F1 ({best['pred_size']:.1f} nm | EE {best['pred_EE']:.1f}%)</div>

                <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:rgba(0,210,140,0.45);letter-spacing:2px;margin-bottom:12px">MATERIALS</div>
                <div class="p-step"><div class="p-num">M</div>
                <div class="p-text">PLGA {best['polymer_MW']:.0f} kDa (LA/GA {best['LA/GA']:.2f}) &nbsp;·&nbsp;
                Organic solvent (acetonitrile or acetone) &nbsp;·&nbsp; Surfactant (HLB {best['surfactant_HLB']:.1f})
                &nbsp;·&nbsp; Deionized water (pH {best['pH']:.2f}) &nbsp;·&nbsp; Drug compound</div></div>

                <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:rgba(0,210,140,0.45);letter-spacing:2px;margin:14px 0 10px">PROCEDURE</div>
                <div class="p-step"><div class="p-num">1</div>
                <div class="p-text"><strong>Organic phase:</strong> Dissolve PLGA {best['polymer_MW']:.0f} kDa in organic solvent. Add drug at drug/polymer = <strong>{best['drug/polymer']:.4f}</strong>. Ensure complete dissolution (sonicate briefly if needed).</div></div>
                <div class="p-step"><div class="p-num">2</div>
                <div class="p-text"><strong>Aqueous phase:</strong> Prepare surfactant at <strong>{best['surfactant_concentration']:.3f}%</strong> (w/v). Adjust pH to <strong>{best['pH']:.2f}</strong> using 0.1 M HCl or NaOH.</div></div>
                <div class="p-step"><div class="p-num">3</div>
                <div class="p-text"><strong>Nanoprecipitation:</strong> Add organic phase dropwise to aqueous phase (Aq/Org ratio = <strong>{best['aqueous/organic']:.2f}</strong>) under continuous magnetic stirring at 600–800 RPM, room temperature.</div></div>
                <div class="p-step"><div class="p-num">4</div>
                <div class="p-text"><strong>Solvent removal:</strong> Stir nanosuspension for 3–4 h at RT under fume hood, or use rotary evaporation at 30°C until organic solvent is fully removed.</div></div>
                <div class="p-step"><div class="p-num">5</div>
                <div class="p-text"><strong>Purification:</strong> Ultracentrifuge at 15,000 × g, 30 min, 4°C. Discard supernatant. Resuspend in deionized water. Repeat ×3.</div></div>
                <div class="p-step"><div class="p-num">6</div>
                <div class="p-text"><strong>Lyophilization:</strong> Freeze-dry with 5% trehalose cryoprotectant. Pre-freeze at −80°C, lyophilize 48 h. Store at −20°C.</div></div>
                <div class="p-step"><div class="p-num">7</div>
                <div class="p-text"><strong>Characterization:</strong> DLS — size &lt; {target_size} nm, PDI &lt; 0.25, Zeta &gt; |±20 mV|. EE% by UV-Vis spectrophotometry post membrane separation. Confirm morphology by TEM.</div></div>
            </div>
            """, unsafe_allow_html=True)

        with col_p2:
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;color:#e8f4ff;margin-bottom:12px">
                All Recommendations
            </div>""", unsafe_allow_html=True)
            for i, row in recs.iterrows():
                bc = "#00d28c" if i == 0 else "rgba(255,255,255,0.08)"
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid {bc};border-radius:12px;padding:12px 14px;margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <span style="font-family:'Syne',sans-serif;font-weight:700;color:#e8f4ff">F{i+1}</span>
                        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#00d28c">
                            {row['pred_size']:.1f} nm · EE {row['pred_EE']:.1f}%
                        </span>
                    </div>
                    <div style="font-size:0.73rem;color:rgba(150,185,220,0.5);line-height:1.9">
                        PLGA {row['polymer_MW']:.0f} kDa &nbsp;·&nbsp; LA/GA {row['LA/GA']:.2f} &nbsp;·&nbsp; D/P {row['drug/polymer']:.4f}<br>
                        Surf. {row['surfactant_concentration']:.3f}% (HLB {row['surfactant_HLB']:.1f}) &nbsp;·&nbsp;
                        Aq/Org {row['aqueous/organic']:.2f} &nbsp;·&nbsp; pH {row['pH']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            proto_df = recs[['drug/polymer','polymer_MW','LA/GA','surfactant_concentration',
                              'surfactant_HLB','aqueous/organic','pH','pred_size','pred_EE']].copy()
            proto_df.insert(0, 'Formulation', [f'F{i+1}' for i in range(len(recs))])
            proto_df.columns = ['Formulation','Drug/Polymer','PLGA MW (kDa)','LA/GA',
                                 'Surf. Conc (%)','Surf. HLB','Aq/Org','pH',
                                 'Pred. Size (nm)','Pred. EE (%)']
            st.download_button("📥  Download Protocol CSV",
                proto_df.to_csv(index=False),
                f"nanoformula_protocol_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    NANOFORMULA AI &nbsp;·&nbsp; PLGA FORMULATION OPTIMIZER &nbsp;·&nbsp;
    R² = 0.88 · MAE ±22 nm &nbsp;·&nbsp; 433 TRAINING FORMULATIONS &nbsp;·&nbsp;
    HARDIK SOOD &nbsp;·&nbsp; IIT (BHU) VARANASI
</div>
""", unsafe_allow_html=True)

