"""Export the source data of Figure 1 (v2) and Supplementary Table 1 (new) from the study results.

Requires the study analysis outputs (results/v5_fig1v2/, produced by fig1v2_analysis.py) and the v5 Figure 1
outcome tables. Writes ../source_data/*.csv, ../source_data/Fig1_source_data.xlsx and
../supplementary/Supplementary_Table_1_*.csv.
"""
import os, json, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'source_data')
SUP = os.path.join(HERE, '..', 'supplementary')
RES = '/home/ubuntu/work/v5/run/results/v5_fig1v2/'
V5F1 = '/home/ubuntu/work/v5/deliver/v5_submission/03_Main_Figures/Figure1/source_data/'
ATLAS = '/home/ubuntu/work/v5/run/results/v5_fig6/SFig_imaging_atlas.csv'
os.makedirs(OUT, exist_ok=True)
os.makedirs(SUP, exist_ok=True)

F = json.load(open(RES + 'F1_facts.json'))

# a: cohort and data layers
a = pd.DataFrame([
    ('n_ukb', 502411, 'UK Biobank participants'),
    ('n_proteomics', 52995, 'Baseline Olink Explore 3072 proteomics in the study dataset (protein file = age gap file, one to one)'),
    ('n_proteins', 2923, 'Unique proteins on Olink Explore 3072'),
    ('n_gap', 52995, 'Participants with 12 organ age gaps'),
    ('n_cohort', F['n_cohort'], 'Analytic cohort, CKM stages 0-3'),
    ('fu_median_years', F['fu_median'], 'Median follow-up, years'),
    ('fu_q1_years', F['fu_q1'], 'Follow-up, 25th percentile'),
    ('fu_q3_years', F['fu_q3'], 'Follow-up, 75th percentile'),
    ('person_years', F['person_years'], 'Person-years of follow-up'),
    ('deaths', F['deaths'], 'Deaths during follow-up'),
    ('end_of_follow_up', '2021-09-30', 'End of hospital inpatient record coverage'),
    ('n_imaging_any', F['n_imaging_any'], 'Cardiac or brain MRI at the imaging visit'),
    ('n_cmr', F['n_cmr'], 'Cardiac MRI'),
    ('n_brain', F['n_brain'], 'Brain MRI'),
    ('n_aortic', F['n_aortic'], 'Aortic distensibility (MRI)'),
    ('n_asi', F['n_asi'], 'Arterial stiffness index (baseline)'),
], columns=['key', 'value', 'description'])
a.to_csv(os.path.join(OUT, 'Fig1a_cohort_data_layers.csv'), index=False)

# b: clocks
b = pd.read_csv(RES + 'F1b_clocks.csv')
desc = {'Organismal': 'Proteins not enriched in any organ', 'Conventional': 'All measured proteins'}
b['protein_set'] = [desc.get(c, f'{c}-enriched proteins (GTEx, >=4-fold)') for c in b.clock]
b.to_csv(os.path.join(OUT, 'Fig1b_organ_clocks.csv'), index=False)

# c: stage profile
c = pd.read_csv(RES + 'F1c_stage_profile.csv')
c['label'] = ['No CKM risk factors', 'Excess or dysfunctional adiposity', 'Metabolic risk factors or CKD',
              'Very-high-risk CKD or PREVENT >=20%']
c.to_csv(os.path.join(OUT, 'Fig1c_stage_profile.csv'), index=False)

# d: structure of the clocks
for f in ['F1d_corr_pearson.csv', 'F1d_corr_spearman.csv', 'F1d_corr_stage0.csv', 'F1d_pca.csv',
          'F1d_independence.csv', 'F1d_linkage.csv']:
    shutil.copy(RES + f, os.path.join(OUT, f.replace('F1d', 'Fig1d')))
P = pd.read_csv(RES + 'F1d_corr_pearson.csv', index_col=0)
org = [o for o in P.index if o not in ('Organismal', 'Conventional')]
iu = np.triu_indices(len(org), 1)
oo = P.loc[org, org].to_numpy()[iu]
ow = P.loc[org, ['Organismal', 'Conventional']].to_numpy().ravel()
pd.DataFrame([dict(pair_type='organ-organ', n_pairs=len(oo), r_median=np.median(oo), r_min=oo.min(), r_max=oo.max()),
              dict(pair_type='organ-whole-body', n_pairs=len(ow), r_median=np.median(ow), r_min=ow.min(), r_max=ow.max())]
             ).to_csv(os.path.join(OUT, 'Fig1d_pair_types.csv'), index=False)

# e: outcomes
shutil.copy(V5F1 + 'Fig1e_tier12_events.csv', os.path.join(OUT, 'Fig1e_tier12_events.csv'))
shutil.copy(V5F1 + 'Fig1e_tier3_events.csv', os.path.join(OUT, 'Fig1e_tier3_events.csv'))
A = pd.read_csv(ATLAS)
im = A.groupby('source').agg(n_phenotypes=('field', 'nunique'), n_max=('n', 'max')).reset_index()
im.to_csv(os.path.join(OUT, 'Fig1e_imaging_phenotypes.csv'), index=False)

# f: evidence chain. Every thumbnail reuses the source data of Figures 2-6 (no new analysis).
MF = '/home/ubuntu/work/v5/deliver/v5_submission/03_Main_Figures/'
F6 = '/home/ubuntu/work/v5/figure6/source_data/'
TGT = ['Kidney', 'Artery', 'Heart', 'Liver', 'Pancreas', 'Brain']
ORGC = ['Kidney', 'Artery', 'Heart', 'Liver', 'Pancreas', 'Brain', 'Immune', 'Lung', 'Intestine', 'Muscle']
PAIR = {'HF': 'Heart', 'CKD': 'Kidney', 'DEM': 'Brain', 'ASCVD': 'Artery', 'T2D': 'Pancreas', 'CLD': 'Liver'}
# Q1 (Fig. 2a): adjusted (M2) age gap differences versus stage 0, target organs
q1 = pd.read_csv(MF + 'Figure2/source_data/Fig2a_stage_differences_M2.csv')
q1 = q1[q1.clock.isin(TGT)].copy()
q1 = pd.concat([pd.DataFrame(dict(clock=TGT, model='M2', stage=0, beta_sd=0.0, lo_sd=0.0, hi_sd=0.0)), q1])
q1['source'] = 'Figure2/source_data/Fig2a_stage_differences_M2.csv (stage 0 = reference)'
q1.sort_values(['clock', 'stage']).to_csv(os.path.join(OUT, 'Fig1f_q1_stage_differences.csv'), index=False)
q1t = pd.read_csv(MF + 'Figure2/source_data/Fig2a_asynchrony_test.csv')
q1t['source'] = 'Figure2/source_data/Fig2a_asynchrony_test.csv'
q1t.to_csv(os.path.join(OUT, 'Fig1f_q1_asynchrony_test.csv'), index=False)
# Q2 (Fig. 3): HR per s.d. of the ten organ clocks for the six tier-1 outcomes (model b = raw age gaps)
q2 = pd.read_csv(MF + 'Figure3/source_data/Fig3e_all_clocks_raw_resid_hr.csv')
q2 = q2[(q2.version == 'raw') & q2.outcome.isin(list(PAIR)) & q2.clock.isin(ORGC)].copy()
q2['paired'] = [PAIR[o] == c for o, c in zip(q2.outcome, q2.clock)]
q2['rank_among_10'] = q2.groupby('outcome').hr.rank(ascending=False).astype(int)
q2['source'] = 'Figure3/source_data/Fig3e_all_clocks_raw_resid_hr.csv (version raw = model b)'
q2 = q2[['outcome', 'clock', 'paired', 'hr', 'lo', 'hi', 'p', 'n', 'events', 'rank_among_10', 'source']]
chk = pd.read_csv(MF + 'Figure3/source_data/Fig3c_disease_correspondence.csv').set_index('outcome')
for o in PAIR:
    assert int(q2[(q2.outcome == o) & q2.paired].rank_among_10.iloc[0]) == int(chk.loc[o, 'rank_among_10_raw'])
q2.to_csv(os.path.join(OUT, 'Fig1f_q2_specificity.csv'), index=False)
# Q3 (Fig. 4b): 10-year CKM composite risk by stage group and number of aged target organs; TOST equivalence
q3 = pd.read_csv(MF + 'Figure4/source_data/Fig4b_dose_response_10y_risk.csv')
q3['source'] = 'Figure4/source_data/Fig4b_dose_response_10y_risk.csv'
q3.to_csv(os.path.join(OUT, 'Fig1f_q3_risk_stage_burden.csv'), index=False)
q3e = pd.read_csv(MF + 'Figure4/source_data/Fig4b_risk_equivalence_composite.csv')
q3e['source'] = 'Figure4/source_data/Fig4b_risk_equivalence_composite.csv'
q3e.to_csv(os.path.join(OUT, 'Fig1f_q3_equivalence.csv'), index=False)
# Q4 (Fig. 5a): cross-fitted gain in C-index over models G and D
q4 = pd.read_csv(MF + 'Figure5/source_data/Fig5a_discrimination_gain.csv')
q4['source'] = 'Figure5/source_data/Fig5a_discrimination_gain.csv'
q4.to_csv(os.path.join(OUT, 'Fig1f_q4_delta_c.csv'), index=False)
# Q5 (Fig. 6a): observed versus PREVENT-predicted 10-year total CVD risk by burden, with O/E and bootstrap 95% CI
# (figure6 v2 export; final Fig. 6a and Results)
q5 = pd.read_csv(F6 + 'Fig6a_oe_by_burden.csv')
q5['source'] = 'figure6/source_data/Fig6a_oe_by_burden.csv'
q5.to_csv(os.path.join(OUT, 'Fig1f_q5_observed_predicted.csv'), index=False)
old = os.path.join(OUT, 'Fig1f_roadmap_findings.csv')
if os.path.exists(old):
    os.remove(old)

# Supplementary Table 1 parts
for src, dst in [('ST1a_selection.csv', 'Supplementary_Table_1a_selection.csv'),
                 ('ST1b_analytic_samples.csv', 'Supplementary_Table_1b_analytic_samples.csv'),
                 ('ST1c_included_excluded.csv', 'Supplementary_Table_1d_included_excluded.csv'),
                 ('ST1d_staging_missingness.csv', 'Supplementary_Table_1e_staging_missingness.csv')]:
    shutil.copy(RES + src, os.path.join(SUP, dst))

# workbook
files = sorted(x for x in os.listdir(OUT) if x.endswith('.csv'))
with pd.ExcelWriter(os.path.join(OUT, 'Fig1_source_data.xlsx')) as w:
    idx = []
    for x in files:
        d = pd.read_csv(os.path.join(OUT, x))
        sh = x[:-4][:31]
        d.to_excel(w, sheet_name=sh, index=False)
        idx.append(dict(sheet=sh, panel=x.split('_')[0].replace('Fig1', ''), file=x, rows=len(d), columns=d.shape[1]))
    pd.DataFrame(idx).to_excel(w, sheet_name='index', index=False)
print(pd.DataFrame(idx).to_string())
print(pd.read_csv(os.path.join(OUT, 'Fig1d_pair_types.csv')).round(3).to_string())
print(im.to_string())
