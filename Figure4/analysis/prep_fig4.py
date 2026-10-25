"""Export source data for Figure 4 (multi-organ ageing burden and disease risk within CKM stages).
Inputs: results/v3/M3_*.csv (module 3) and results/v4/F4_*.csv (fig4_reanalysis.py)."""
import os, glob
import pandas as pd
R3 = './results/v3/'
R4 = './results/v4/'
OUT = './v4/Figure4/source_data/'
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(OUT + '*'):
    os.remove(f)
OUTS = ['CKMC', 'HF', 'ASCVD', 'CKD', 'T2D', 'CLD', 'DEM']
GLAB = {'S0/B0': 'Stage 0, 0 aged organs', 'S0/B1': 'Stage 0, 1 aged organ', 'S0/B>=2': 'Stage 0, >=2 aged organs',
        'S1/B0': 'Stage 1, 0 aged organs', 'S1/B1': 'Stage 1, 1 aged organ', 'S1/B>=2': 'Stage 1, >=2 aged organs',
        'S2-3/B0': 'Stage 2-3, 0 aged organs', 'S2-3/B1': 'Stage 2-3, 1 aged organ', 'S2-3/B>=2': 'Stage 2-3, >=2 aged organs'}

# a  standardised cumulative incidence of the CKM composite within stage
c = pd.read_csv(R4 + 'F4_std_cuminc_composite.csv'); c['group_label'] = c.group.map(GLAB)
c.rename(columns={'t': 'years', 'cuminc': 'std_cumulative_incidence', 'lo': 'ci_lo', 'hi': 'ci_hi'}).to_csv(
    OUT + 'Fig4a_std_cuminc_composite.csv', index=False)
r = pd.read_csv(R4 + 'F4_at_risk_composite.csv'); r['group_label'] = r.group.map(GLAB)
r.rename(columns={'t': 'years'}).to_csv(OUT + 'Fig4a_numbers_at_risk.csv', index=False)
w = pd.read_csv(R4 + 'F4_within_stage_hr_composite.csv')
w['contrast'] = w.term.map({'b1': '1 vs 0 aged organs', 'b2': '>=2 vs 0 aged organs'})
w[['stage', 'contrast', 'hr', 'lo', 'hi', 'p', 'n', 'events']].to_csv(OUT + 'Fig4a_within_stage_hr.csv', index=False)

# b  dose-response and cross-stage comparison
pd.read_csv(R4 + 'F4_dose_response_risk10.csv').to_csv(OUT + 'Fig4b_dose_response_10y_risk.csv', index=False)
t = pd.read_csv(R4 + 'F4_dose_response_hr.csv')
t['contrast'] = t.term.map({'n1': '1 vs 0', 'n2': '2 vs 0', 'n3': '>=3 vs 0', 'b4': 'per aged organ (0, 1, 2, >=3)'})
t[['outcome', 'model', 'contrast', 'hr', 'lo', 'hi', 'p', 'n', 'events']].to_csv(OUT + 'Fig4b_dose_response_hr.csv', index=False)
e = pd.read_csv(R3 + 'M3_risk_equivalence.csv')
e[e.outcome == 'CKMC'].to_csv(OUT + 'Fig4b_risk_equivalence_composite.csv', index=False)

# c  outcome-specific associations and absolute risk differences by stage
pd.read_csv(R4 + 'F4_outcome_specific.csv').to_csv(OUT + 'Fig4c_outcome_hr_interaction.csv', index=False)
pd.read_csv(R4 + 'F4_risk_difference_by_stage.csv').to_csv(OUT + 'Fig4c_risk_difference_by_stage.csv', index=False)

# d  specification analysis
pd.read_csv(R4 + 'F4_specification_composite.csv').to_csv(OUT + 'Fig4d_specification_composite.csv', index=False)

# e  spread to multisystem disease
m = pd.read_csv(R3 + 'M3_multistate.csv')
m = m[m.term.isin(['b1', 'b2', 'st1', 'st2', 'st3'])]
m[['transition', 'exposure', 'hr', 'lo', 'hi', 'p', 'n', 'events']].to_csv(OUT + 'Fig4e_multistate_hr.csv', index=False)
o = pd.read_csv(R4 + 'F4_state_occupation_12y.csv'); o['group_label'] = o.group.map(GLAB)
o.to_csv(OUT + 'Fig4e_state_occupation_12y.csv', index=False)
pd.read_csv(R3 + 'M3_rmst_multimorbidity.csv').to_csv(OUT + 'Fig4e_rmst_multisystem_free.csv', index=False)

# f  stage-associated risk shared with organ age gaps
d = pd.read_csv(R3 + 'M3_decomposition.csv')
d[d.outcome.isin(OUTS)].to_csv(OUT + 'Fig4f_stage_hr_attenuation.csv', index=False)
print(sorted(os.listdir(OUT)))
