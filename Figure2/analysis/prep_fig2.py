"""Export source data for Figure 2 (values plotted in each panel; no individual-level data).
Inputs: results/L1_*.csv and L2_*.csv (02_layer1.py, 03_layer2.py), results/v4/F2_*.csv (fig2_reanalysis.py,
fig2_roc.py) and Supplementary/fig2_text_checks/*.csv (fig2_text_checks.py)."""
import os, json, glob
import pandas as pd
B = './results/'
RV = B + 'v4/'
CHK = './v4/Supplementary/fig2_text_checks/'
OUT = './v4/Figure2/source_data/'
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(OUT + '*.csv') + glob.glob(OUT + '*.xlsx'):
    os.remove(f)


def w(df, name):
    df.to_csv(OUT + name, index=False)


L = pd.read_csv(B + 'L1_stage_organ.csv').rename(columns={'organ': 'clock'})
L['stage'] = L.term.str[-1].astype(int)
L['beta_years'] = L.beta
COLS = ['clock', 'model', 'stage', 'beta_sd', 'lo_sd', 'hi_sd', 'beta_years', 'p', 'q', 'n']

# a  stage trajectories under M2, kidney-function-residualised trajectories, onset shares and asynchrony test
w(L[L.model == 'M2'][COLS], 'Fig2a_stage_differences_M2.csv')
k = pd.read_csv(RV + 'F2_kidney_residualised_stage.csv').rename(columns={'lo': 'lo_sd', 'hi': 'hi_sd'})
w(k[['clock', 'stage', 'beta_sd', 'lo_sd', 'hi_sd', 'p', 'q']], 'Fig2a_kidney_residualised.csv')
w(pd.read_csv(RV + 'F2_onset_share.csv'), 'Fig2a_onset_share.csv')
s = json.load(open(RV + 'F2_summary.json'))
w(pd.DataFrame([{'n': s['n'], 'bootstrap_samples': s['n_boot'], 'wald_chi2_equal_onset_share': s['wald_equal_share1'],
                 'df': s['wald_df'], 'p': s['wald_p'], 'late_rising_clocks': ', '.join(s['late_clocks']),
                 'late_group_reproduced_in_bootstrap': s['cluster_reproduced'],
                 'late_minus_early_mean_onset_share': s['late_minus_early_share1'],
                 'lo': s['late_minus_early_lo'], 'hi': s['late_minus_early_hi']}]), 'Fig2a_asynchrony_test.csv')
# b  Shapley decomposition of age-gap variance by six CKM-related trait domains
w(pd.read_csv(B + 'L1_shapley.csv').rename(columns={'organ': 'clock', 'total': 'total_r2'}), 'Fig2b_shapley.csv')
# c  stage differences under M2 and M3
w(L[L.model.isin(['M2', 'M3'])][COLS], 'Fig2c_stage_differences_M2_M3.csv')
# d  Gelbach decomposition of the M2 - M3 change (five staging domains), stages 1 and 3
g = pd.read_csv(RV + 'F2_gelbach.csv')
g = g[(g.level == 'domain') & (g.stage.isin([1, 3]))].drop(columns='level').rename(columns={'component': 'domain'})
w(g, 'Fig2d_gelbach_domains.csv')
# e  overlap of age gaps across stages
w(pd.read_csv(RV + 'F2_within_stage.csv'), 'Fig2e_within_stage.csv')
den = pd.read_csv(RV + 'F2_density.csv')
w(den[den.clock.isin(['Kidney', 'Heart'])], 'Fig2e_density.csv')
# f  burden distribution, odds ratios, kidney-function-residualised distribution
D = pd.read_csv(B + 'L2_burden_dist.csv')
D.columns = ['stage', 'organs_0', 'organs_1', 'organs_2', 'organs_3plus']
w(D.merge(pd.read_csv(CHK + 'burden_counts.csv'), on='stage'), 'Fig2f_burden_distribution.csv')
rows = []
for m, f in [('M2', 'L2_or_burden2.csv'), ('M3', 'L2_or_burden2_M3.csv')]:
    d = pd.read_csv(B + f, index_col=0)
    for kk, r in d.iterrows():
        rows.append((m, int(kk[-1]), r.OR, r.lo, r.hi, r.p))
w(pd.DataFrame(rows, columns=['model', 'stage', 'OR', 'lo', 'hi', 'p']), 'Fig2f_odds_high_burden.csv')
R = pd.read_csv(CHK + 'resid_burden_dist.csv')
R.columns = ['stage', 'organs_0', 'organs_1', 'organs_2', 'organs_3plus']
R['ge2_pct'] = 100 * (R.organs_2 + R.organs_3plus)
w(R, 'Fig2f_burden_kidney_residualised.csv')
# g  identification of high burden in stages 1-2
ren = {'Routine clinical panel without eGFR': '+ clinical panel without eGFR',
       '+ routine clinical panel': '+ eGFR (full clinical panel)'}
A = pd.read_csv(CHK + 'auc_ci.csv')
A['model'] = A.model.replace(ren)
w(A, 'Fig2g_auc.csv')
w(pd.read_csv(RV + 'F2_roc.csv'), 'Fig2g_roc.csv')
w(pd.read_csv(B + 'L2_smd_highburden.csv'), 'Fig2g_smd.csv')
w(pd.read_csv(RV + 'F2_prevent_categories.csv'), 'Fig2g_prevent_categories.csv')
print(sorted(os.listdir(OUT)))
# Supplementary Fig. 2d: M2 stage differences after residualising age gaps on eGFR and cystatin C
SUP = './v4/Supplementary/Supplementary_Figures/source_data/'
k = pd.read_csv(RV + 'F2_kidney_residualised_stage.csv')
k = k.rename(columns={'clock': 'organ', 'lo': 'lo_sd', 'hi': 'hi_sd'})
k['term'] = 'st' + k.stage.astype(str)
k['model'] = 'M2, kidney-function residualised'
k[['organ', 'model', 'term', 'beta_sd', 'lo_sd', 'hi_sd', 'p', 'q', 'share1', 'share1_lo', 'share1_hi']].to_csv(
    SUP + 'SFig2d_kidney_residualised.csv', index=False)
