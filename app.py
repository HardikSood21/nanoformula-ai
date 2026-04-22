import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

# Use a clean, professional plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="NanoOptimizer | IIT BHU",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS (SCIENTIFIC THEME) ==========
st.markdown("""
<style>
    /* Main Background and Text */
    .reportview-container { background: #fdfdfd; }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #1E3A8A;
        text-align: left;
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    .institution-sub {
        font-family: 'Georgia', serif;
        font-style: italic;
        color: #4B5563;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1E3A8A;
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #F8FAFC; }
    
    /* Buttons */
    .stButton>button {
        border-radius: 5px;
        border: none;
        background-color: #1E3A8A;
        color: white;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<h1 class="main-header">Nanoparticle Formulation Optimizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="institution-sub">Pharmaceutical Engineering & Technology, Indian Institute of Technology (BHU) Varanasi<br>Lab of Dr. Ruchi Chawla</p>', unsafe_allow_html=True)

# ========== DATA LOADING (LOGIC UNCHANGED) ==========
@st.cache_resource
def load_all_models():
    models = {}
    # PLGA models
    try:
        with open('model_particle_size_final.pkl', 'rb') as f: models['plga_size'] = pickle.load(f)
        with open('model_ee_final.pkl', 'rb') as f: models['plga_ee'] = pickle.load(f)
        models['plga_data'] = pd.read_csv('PLGA_nanoparticles_dataset.csv')
        models['plga_loaded'] = True
    except: models['plga_loaded'] = False
    
    # Chitosan models
    try:
        with open('model_chitosan_size.pkl', 'rb') as f: models['chitosan_size'] = pickle.load(f)
        with open('chitosan_features.pkl', 'rb') as f: models['chitosan_features'] = pickle.load(f)
        models['chitosan_data'] = pd.read_csv('chitosan_nanoparticles_dataset.csv')
        models['chitosan_loaded'] = True
    except: models['chitosan_loaded'] = False
    return models

models = load_all_models()

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://img.icons8.com/external-flat-icons-inmotus-design/64/000000/external-Atom-science-flat-icons-inmotus-design-3.png", width=60)
    st.header("Control Panel")
    
    polymer_options = []
    if models.get('plga_loaded'): polymer_options.append("PLGA System")
    if models.get('chitosan_loaded'): polymer_options.append("Chitosan System")
    
    if not polymer_options:
        st.error("Model files missing.")
        st.stop()
        
    selected_polymer = st.radio("Select Delivery Platform", polymer_options)
    st.markdown("---")

# ========== PLGA WORKFLOW ==========
if "PLGA" in selected_polymer:
    tab1, tab2 = st.tabs(["🎯 Optimization Engine", "📚 Model Performance"])
    
    with tab1:
        st.markdown("### Drug-Polymer Input Parameters")
        
        # Grid layout for inputs
        col_a, col_b = st.columns(2)
        with col_a:
            mol_MW = st.number_input("Molecular Weight (g/mol)", 100.0, 1000.0, 420.5)
            mol_logP = st.number_input("LogP (Lipophilicity)", -5.0, 10.0, 3.1)
            mol_TPSA = st.number_input("TPSA (Å²)", 0.0, 300.0, 72.0)
            mol_mp = st.number_input("Melting Point (°C)", 0.0, 500.0, 175.0)
        with col_b:
            mol_Hacc = st.number_input("H-Acceptors", 0, 20, 5)
            mol_Hdon = st.number_input("H-Donors", 0, 10, 2)
            mol_het = st.number_input("Heteroatoms", 0, 30, 7)
            
        with st.expander("Polymer Specifics & Targets"):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                polymer_mode = st.selectbox("PLGA Constraint", ["Auto-optimize", "Fixed Specification"])
                target_size = st.slider("Target Size (nm)", 50, 250, 180)
            with p_col2:
                n_recs = st.slider("Result Count", 3, 10, 5)
                min_ee = st.slider("Min EE (%)", 50, 95, 70)

        if st.button("RUN OPTIMIZATION ALGORITHM", use_container_width=True):
            # ... (Your existing PLGA logic remains exactly the same) ...
            drug_properties = {
                'mol_MW': mol_MW, 'mol_logP': mol_logP, 'mol_TPSA': mol_TPSA,
                'mol_melting_point': mol_mp, 'mol_Hacceptors': mol_Hacc,
                'mol_Hdonors': mol_Hdon, 'mol_heteroatoms': mol_het
            }
            df_plga = models['plga_data']
            successful = df_plga[df_plga['particle_size'] < target_size]
            if len(successful) < 20: successful = df_plga[df_plga['particle_size'] < 250]
            
            formulation_space = {
                'polymer_MW': successful['polymer_MW'].unique().tolist(),
                'LA/GA': successful['LA/GA'].unique().tolist(),
                'drug/polymer': np.linspace(successful['drug/polymer'].quantile(0.1), successful['drug/polymer'].quantile(0.9), 40).tolist(),
                'surfactant_concentration': np.linspace(successful['surfactant_concentration'].min(), successful['surfactant_concentration'].quantile(0.9), 25).tolist(),
                'surfactant_HLB': successful['surfactant_HLB'].unique().tolist(),
                'aqueous/organic': successful['aqueous/organic'].unique().tolist(),
                'pH': successful['pH'].unique().tolist(),
                'solvent_polarity_index': successful['solvent_polarity_index'].unique().tolist()
            }
            
            random.seed(42)
            np.random.seed(42)
            candidates = []
            for _ in range(20000):
                candidate = {**drug_properties, 
                            'polymer_MW': random.choice(formulation_space['polymer_MW']),
                            'LA/GA': random.choice(formulation_space['LA/GA']),
                            'drug/polymer': random.choice(formulation_space['drug/polymer']),
                            'surfactant_concentration': random.choice(formulation_space['surfactant_concentration']),
                            'surfactant_HLB': random.choice(formulation_space['surfactant_HLB']),
                            'aqueous/organic': random.choice(formulation_space['aqueous/organic']),
                            'pH': random.choice(formulation_space['pH']),
                            'solvent_polarity_index': random.choice(formulation_space['solvent_polarity_index'])}
                candidates.append(candidate)
            
            cand_df = pd.DataFrame(candidates)
            plga_features = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index']
            cand_df['pred_size'] = models['plga_size'].predict(cand_df[plga_features])
            cand_df['pred_EE'] = models['plga_ee'].predict(cand_df[plga_features])
            
            cand_df['size_score'] = np.where(cand_df['pred_size'] <= target_size, (target_size - cand_df['pred_size']) / target_size, -0.5)
            cand_df['ee_score'] = (cand_df['pred_EE'] - 50) / 50
            cand_df['total_score'] = 0.7 * cand_df['size_score'] + 0.3 * cand_df['ee_score']
            
            top = cand_df.nlargest(n_recs * 3, 'total_score')
            recs = top.head(n_recs).sort_values('pred_size').reset_index(drop=True)

            # RESULTS DISPLAY
            st.markdown("---")
            st.subheader("Predicted Optimal Formulations")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Search Space", "20k trials")
            m2.metric("Target Achievement", f"{(cand_df['pred_size'] < target_size).sum()}")
            m3.metric("Optimum Size", f"{recs['pred_size'].min():.1f} nm")
            m4.metric("Optimum EE", f"{recs['pred_EE'].max():.1f}%")

            st.dataframe(recs[['drug/polymer', 'polymer_MW', 'LA/GA', 'surfactant_concentration', 'pred_size', 'pred_EE']].style.background_gradient(cmap='Blues'))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig, ax = plt.subplots()
                sns.barplot(x=recs.index, y=recs['pred_size'], palette="Blues_d", ax=ax)
                ax.axhline(target_size, color='red', ls='--')
                ax.set_title("Predicted Particle Size by Formulation")
                st.pyplot(fig)
            with col_p2:
                fig, ax = plt.subplots()
                sns.scatterplot(data=recs, x='pred_size', y='pred_EE', size='total_score', hue='total_score', palette='viridis', ax=ax)
                ax.set_title("Size vs. Entrapment Efficiency")
                st.pyplot(fig)

    with tab2:
        st.info("**Experimental Validation:** Model trained on 433 multi-drug formulations. R² = 0.88 for Particle Size.")
        st.table(pd.DataFrame({
            "Metric": ["R-Squared", "MAE", "RMSE", "Data Source"],
            "Value": ["0.88", "22.4 nm", "29.1 nm", "IIT BHU Repository"]
        }))

# ========== CHITOSAN WORKFLOW ==========
elif "Chitosan" in selected_polymer:
    tab1, tab2 = st.tabs(["🎯 Optimization Engine", "🧪 Lab Protocol & Model Info"])
    
    with tab1:
        st.markdown("### Chitosan-TPP Ionotropic Gelation Optimizer")
        st.warning("⚠️ **Note:** The Chitosan model currently supports size prediction for blank nanoparticles only based on 44 formulations.")
        
        # Grid layout for inputs
        col_a, col_b = st.columns(2)
        with col_a:
            chitosan_mode = st.selectbox("Chitosan Selection", ["Auto-optimize (find best type)", "I have specific chitosan"])
            if chitosan_mode == "I have specific chitosan":
                chitosan_mw = st.selectbox("Chitosan MW (kDa)", [5, 20, 50, 310], 
                                           format_func=lambda x: f"{x} kDa ({'HMW' if x==310 else 'LMW' if x==50 else f'{x} kDa'})")
                fix_mw = True
            else:
                chitosan_mw = None
                fix_mw = False
                
        with col_b:
            target_size_cs = st.slider("Target Size (nm)", 50, 250, 150, 10, key="cs_size")
            n_recs_cs = st.slider("# Recommendations", 3, 10, 5, 1, key="cs_recs")

        if st.button("RUN CHITOSAN OPTIMIZATION", use_container_width=True):
            with st.spinner('🔬 Running Monte Carlo simulation for Chitosan...'):
                chitosan_features = models['chitosan_features']
                
                # Search space
                if fix_mw:
                    mw_options = [chitosan_mw]
                else:
                    mw_options = [5, 20, 50, 310]
                    
                conc_options = np.linspace(0.10, 1.0, 25).tolist()
                tpp_options = np.linspace(0.10, 1.0, 25).tolist()
                
                random.seed(42)
                np.random.seed(42)
                
                n_candidates = 10000
                candidates = []
                
                for _ in range(n_candidates):
                    mw = random.choice(mw_options)
                    conc = random.choice(conc_options)
                    tpp = random.choice(tpp_options)
                    
                    candidate = {
                        'chitosan_MW': mw,
                        'chitosan_conc': conc,
                        'TPP_conc': tpp,
                        'chitosan_TPP_ratio': conc / tpp,
                        'conc_x_TPP': conc * tpp,
                        'MW_x_conc': mw * conc,
                        'MW_x_TPP': mw * tpp,
                        'log_MW': np.log10(mw),
                        'total_solute': conc + tpp,
                        'chitosan_fraction': conc / (conc + tpp)
                    }
                    candidates.append(candidate)
                    
                cand_df = pd.DataFrame(candidates)
                X_cand = cand_df[chitosan_features]
                cand_df['pred_size'] = models['chitosan_size'].predict(X_cand)
                
                # Score
                cand_df['score'] = np.where(
                    cand_df['pred_size'] <= target_size_cs,
                    (target_size_cs - cand_df['pred_size']) / target_size_cs, -0.5)
                    
                # Get diverse recommendations
                top = cand_df.nlargest(n_recs_cs * 3, 'score')
                diverse = []
                used_mw = set()
                
                for _, row in top.iterrows():
                    mw = row['chitosan_MW']
                    if mw not in used_mw or len(diverse) < 2:
                        diverse.append(row)
                        used_mw.add(mw)
                    if len(diverse) >= n_recs_cs:
                        break
                        
                if len(diverse) < n_recs_cs:
                    diverse = [row for _, row in top.head(n_recs_cs).iterrows()]
                    
                recs_cs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)
                
                # Save best to session state for the Protocol Tab
                st.session_state['best_cs'] = recs_cs.iloc[0]
                
                # RESULTS DISPLAY
                st.markdown("---")
                st.subheader("Predicted Optimal Formulations")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Search Space", f"{n_candidates:,} trials")
                c2.metric(f"Target Achievement (< {target_size_cs}nm)", f"{(cand_df['pred_size'] < target_size_cs).sum():,}")
                c3.metric("Optimum Size", f"{recs_cs['pred_size'].min():.1f} nm")
                
                # Display Dataframe
                mw_type_map = {5: '5 kDa', 20: '20 kDa', 50: 'LMW (~50 kDa)', 310: 'HMW (~310 kDa)'}
                display_cs = recs_cs[['chitosan_MW', 'chitosan_conc', 'TPP_conc', 'chitosan_TPP_ratio', 'pred_size']].copy()
                display_cs['Chitosan Type'] = display_cs['chitosan_MW'].map(mw_type_map)
                display_cs = display_cs[['Chitosan Type', 'chitosan_MW', 'chitosan_conc', 'TPP_conc', 'chitosan_TPP_ratio', 'pred_size']]
                display_cs.columns = ['Chitosan Type', 'MW (kDa)', 'Chitosan Conc (mg/mL)', 'TPP Conc (mg/mL)', 'CS:TPP Ratio', 'Pred. Size (nm)']
                
                st.dataframe(display_cs.style.format({
                    'MW (kDa)': '{:.0f}', 'Chitosan Conc (mg/mL)': '{:.2f}', 'TPP Conc (mg/mL)': '{:.2f}',
                    'CS:TPP Ratio': '{:.2f}', 'Pred. Size (nm)': '{:.1f}'
                }).background_gradient(subset=['Pred. Size (nm)'], cmap='Greens_r'), use_container_width=True)
                
                # Plots with Seaborn matching paper-ready style
                p1, p2 = st.columns(2)
                with p1:
                    fig1, ax1 = plt.subplots()
                    # Custom color palette based on target success
                    colors = ['#10B981' if s < target_size_cs else '#EF4444' for s in recs_cs['pred_size']]
                    sns.barplot(x=recs_cs.index, y=recs_cs['pred_size'], palette=colors, ax=ax1, edgecolor=".2")
                    ax1.axhline(target_size_cs, color='black', ls='--', linewidth=1.5, label=f'Target ({target_size_cs} nm)')
                    ax1.set_xticklabels([f"F{i+1}" for i in range(len(recs_cs))])
                    ax1.set_ylabel("Predicted Size (nm)")
                    ax1.set_title("Predicted Particle Sizes")
                    ax1.legend()
                    st.pyplot(fig1)
                    
                with p2:
                    fig2, ax2 = plt.subplots()
                    sns.scatterplot(data=recs_cs, x='chitosan_conc', y='pred_size', 
                                    hue='chitosan_MW', s=250, palette="viridis", edgecolor="black", ax=ax2)
                    ax2.axhline(target_size_cs, color='red', ls='--', alpha=0.5)
                    ax2.set_xlabel("Chitosan Concentration (mg/mL)")
                    ax2.set_ylabel("Predicted Size (nm)")
                    ax2.set_title("Concentration vs. Size Matrix")
                    st.pyplot(fig2)

    with tab2:
        st.markdown("### 🧪 Standardized Experimental Protocol")
        if 'best_cs' in st.session_state:
            best = st.session_state['best_cs']
            mw_type_map = {5: '5 kDa', 20: '20 kDa', 50: 'LMW (~50 kDa)', 310: 'HMW (~310 kDa)'}
            
            st.info(f"""
            **Optimal Parameters (Predicted Size: {best['pred_size']:.1f} nm)**
            
            **Materials Required:**
            * Chitosan: **{mw_type_map.get(best['chitosan_MW'], '')}** ({best['chitosan_MW']:.0f} kDa)
            * Sodium Tripolyphosphate (TPP)
            * 1% Acetic acid solution (v/v)
            * Deionized water
            
            **Formulation Steps:**
            1. Dissolve chitosan to a final concentration of **{best['chitosan_conc']:.2f} mg/mL** in 1% acetic acid.
            2. Prepare TPP solution at **{best['TPP_conc']:.2f} mg/mL** in deionized water.
            3. Ensure the CS:TPP mass ratio is maintained at exactly **{best['chitosan_TPP_ratio']:.2f}**.
            4. Add TPP solution dropwise to the chitosan solution under constant magnetic stirring (500-700 RPM).
            5. Allow cross-linking to proceed by stirring for 30 minutes at room temperature.
            6. Characterize immediately via DLS for particle size, PDI, and zeta potential.
            """)
        else:
            st.info("👈 Run the Optimization Engine in the previous tab to generate a targeted laboratory protocol.")
            
        st.markdown("---")
        st.markdown("### 📊 Chitosan Model Information")
        st.table(pd.DataFrame({
            "Metric": ["Algorithm", "Training Dataset", "R² Score", "Accuracy Margin"],
            "Value": ["Random Forest", "44 Formulations", "0.83", "±15-20 nm"]
        }))

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9CA3AF; font-size: 12px; font-family: sans-serif;'>
    © 2026 IIT BHU Pharmaceutical Engineering | Developed by Hardik Sood | Supervision: Dr. Ruchi Chawla<br>
    The computational results are predictive and intended for research guidance only.
</div>
""", unsafe_allow_html=True)

