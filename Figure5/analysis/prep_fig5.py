"""Export Source Data for Figure 5 (clinical value of organ age gaps over guideline-based assessment).
Inputs: results/v4/F5_*.csv (fig5_reanalysis.py).
Outputs: Figure5/source_data/*.csv; source data for Supplementary Figs 12-13.
"""
import os
import pandas as pd, numpy as np
ROOT = './'
R4 = ROOT + 'results/v4/'
OUT = ROOT + 'v4/Figure5/source_data/'
SUP = ROOT + 'v4/Supplementary/Supplementary_Figures/source_data/'
os.makedirs(OUT, exist_ok=True)
OLAB = {'CKMC': 'CKM composite', 'HF': 'Heart failure', 'ASCVD': 'ASCVD', 'CKD': 'CKD', 'T2D': 'Type 2 diabetes',
        'CLD': 'Chronic liver disease', 'DEM': 'Dementia'}


def w(df, name, d=OUT):
    df.to_csv(d + name + '.csv', index=False)
    print(name, df.shape)


D = pd.read_csv(R4 + 'F5_disc_decomposition.csv')
D['outcome_label'] = D.outcome.map(OLAB)
# a: discrimination over model G and model D
a = D[D.model.isin(['G+gaps', 'D+gaps'])][['outcome', 'outcome_label', 'baseline', 'n', 'events', 'events10', 'C_base', 'C_new',
                                            'dC', 'lo', 'hi']].copy()
a['baseline'] = a.baseline.map({'G': 'Model G', 'D': 'Model D'})
w(a, 'Fig5a_discrimination_gain')
# b: source of the gain
b = D[D.baseline == 'G'][['outcome', 'outcome_label', 'added', 'paired', 'n', 'events', 'C_base', 'C_new', 'dC', 'lo', 'hi']].copy()
b['ci_excludes_0'] = (b.lo > 0) | (b.hi < 0)
w(b, 'Fig5b_gain_by_clock')
# c: tier-2 diseases, negative control, organ-system composites (two baselines)
B = pd.read_csv(R4 + 'F5_tier2_dC_two_baselines.csv')
c = B[['outcome', 'disease', 'system', 'clock', 'set', 'n_G', 'events_G', 'C_G', 'C_Ggaps', 'dC_G', 'lo_G', 'hi_G', 'p_G', 'q_G',
       'n_min', 'events_min', 'C_min', 'C_mingaps', 'dC_min', 'lo_min', 'hi_min', 'p_min', 'q_min']].copy()
# column order, group, label and tier-1 flag taken from Fig. 3a so that both panels list the diseases identically
o3 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Figure3', 'source_data',
                              'Fig3a_atlas_tier2_model_b.csv')).drop_duplicates('outcome')
c = c.merge(o3[['outcome', 'label', 'column_group', 'column_order', 'tier1']], on='outcome', how='left')
w(c, 'Fig5c_tier2_gain_two_baselines')
# d: case capture in stages 1-2
cc = pd.read_csv(R4 + 'F5_capture_curves.csv'); cc['outcome_label'] = cc.outcome.map(OLAB)
w(cc, 'Fig5d_case_capture_curves')
c10 = pd.read_csv(R4 + 'F5_capture10_diff.csv'); c10['outcome_label'] = c10.outcome.map(OLAB)
w(c10, 'Fig5d_capture_at_10pct')
# e: observed risk among reclassified participants
rc = pd.read_csv(R4 + 'F5_reclass_cells.csv'); rc['outcome_label'] = rc.outcome.map(OLAB)
w(rc, 'Fig5e_reclassification_observed_risk')
# f: difference in net benefit
nb = pd.read_csv(R4 + 'F5_dnb_ci.csv'); nb['outcome_label'] = nb.outcome.map(OLAB)
w(nb, 'Fig5f_net_benefit_difference')
# ---------------------------------------------------------------- supplementary
w(nb[['outcome', 'outcome_label', 'pt', 'nb_G', 'nb_Ggaps', 'nb_treat_all']], 'SFig12a_net_benefit_curves', SUP)
w(c10, 'SFig12b_capture_at_10pct', SUP)
nri = pd.read_csv(R4 + 'F5_nri_ci.csv'); nri['outcome_label'] = nri.outcome.map(OLAB)
w(nri, 'SFig12c_nri', SUP)
w(c[c.set == 'Tier 2'], 'SFig12d_tier2_two_baselines', SUP)
st = pd.read_csv(R4 + 'F5_stage_dC.csv'); st['outcome_label'] = st.outcome.map(OLAB)
w(st, 'SFig13b_stage_specific_gain', SUP)
