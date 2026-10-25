"""Expanded Supplementary Table 3.
3a  Clock characteristics and age-gap diagnostics in the analytic cohort (n = 40,674)
3b  Sensitivity analysis excluding participants with >50% missing Olink protein values
Columns that require the clock-training logs (numbers of candidate/selected proteins,
hold-out r and MAE) are left as author placeholders; nothing is imputed for them.
"""
import sys, os, json
sys.path.insert(0, '/home/ubuntu/work/v5/run/analysis')
os.chdir('/home/ubuntu/work/v5/run')
import numpy as np, pandas as pd
import statsmodels.api as sm
from v3_common import subcohort, ORG, TIER1

BASE = '/home/ubuntu/work/v5/methods_clock/'
RES = BASE + 'results/'
OUT = BASE + 'tables/'
os.makedirs(OUT, exist_ok=True)

LAB = {'Organismal': 'Organismal', 'Conventional': 'Conventional'}
SET = {o: f'{o.lower()}-enriched proteins' for o in ORG}
SET['Immune'] = 'immune-enriched proteins'
SET['Organismal'] = 'proteins not enriched in any organ'
SET['Conventional'] = 'all measured proteins'
ORDER = ['Artery', 'Brain', 'Heart', 'Immune', 'Intestine', 'Kidney', 'Liver', 'Lung', 'Muscle', 'Pancreas',
         'Organismal', 'Conventional']

s = subcohort()
z = pd.read_parquet('/home/ubuntu/work/v5/protein_zero_share.parquet')[['eid', 'zshare']]
s = s.merge(z, on='eid', how='left'); s['hi'] = s.zshare > 0.5
rows = []
for o in ORDER:
    y = s['yr_' + o]
    X = sm.add_constant(np.column_stack([s.age, s.age ** 2]))
    r2 = sm.OLS(y.values, X).fit().rsquared
    mw = y[s.male == 0].mean(); mm = y[s.male == 1].mean()
    rows.append({'Clock': o, 'Protein set': SET[o],
                 'Candidate proteins, n': '[authors]', 'Selected proteins, n': '[authors]',
                 'Hold-out r with age': '[authors]', 'Hold-out MAE, years': '[authors]',
                 '1 s.d. of age gap, years': f'{y.std():.2f}',
                 'Variance explained by age and age², %': f'{100 * r2:.2f}',
                 'Mean age gap, women / men, years': f'{mw:.2f} / {mm:.2f}',
                 's.d., >50% missing / others, years': f'{y[s.hi].std():.2f} / {y[~s.hi].std():.2f}'})
A = pd.DataFrame(rows)
A.to_csv(OUT + 'ST3a_clock_characteristics.csv', index=False)

T1 = pd.read_csv(RES + 'S_missing_tier1.csv')
B = pd.read_csv(RES + 'S_missing_burden.csv')
LABD = {'HF': 'Heart failure', 'ASCVD': 'ASCVD', 'CKD': 'Chronic kidney disease', 'T2D': 'Type 2 diabetes',
        'CLD': 'Chronic liver disease', 'DEM': 'Dementia', 'CKMC': 'CKM composite'}
fmt = lambda r: f'{r.hr:.2f} ({r.lo:.2f}–{r.hi:.2f})'
rows = []
for k, o in TIER1:
    t = T1[T1.outcome == k].set_index('sample')
    rows.append({'Exposure': f'{o} age gap, per s.d.', 'Outcome': LABD[k],
                 'Full cohort, events': f"{int(t.loc['Full cohort', 'events']):,}",
                 'Full cohort, HR (95% CI)': fmt(t.loc['Full cohort']),
                 'Excluding >50% missing, events': f"{int(t.loc['Excluding >50% missing', 'events']):,}",
                 'Excluding >50% missing, HR (95% CI)': fmt(t.loc['Excluding >50% missing']),
                 'Excluding ≥40% missing, events': f"{int(t.loc['Excluding >=40% missing', 'events']):,}",
                 'Excluding ≥40% missing, HR (95% CI)': fmt(t.loc['Excluding >=40% missing']),
                 '>50% missing only, HR (95% CI)': fmt(t.loc['>50% missing only']),
                 'P heterogeneity': f"{t.loc['P heterogeneity', 'p']:.2f}"})
for k in ['CKMC', 'HF', 'ASCVD', 'CKD', 'T2D', 'CLD', 'DEM']:
    b = B[(B.outcome == k) & (B.term == 'b2')].set_index('sample')
    rows.append({'Exposure': '≥2 vs 0 aged organs', 'Outcome': LABD[k],
                 'Full cohort, events': f"{int(b.loc['Full cohort', 'events']):,}",
                 'Full cohort, HR (95% CI)': fmt(b.loc['Full cohort']),
                 'Excluding >50% missing, events': f"{int(b.loc['Excluding >50% missing', 'events']):,}",
                 'Excluding >50% missing, HR (95% CI)': fmt(b.loc['Excluding >50% missing']),
                 'Excluding ≥40% missing, events': f"{int(b.loc['Excluding >=40% missing', 'events']):,}",
                 'Excluding ≥40% missing, HR (95% CI)': fmt(b.loc['Excluding >=40% missing']),
                 '>50% missing only, HR (95% CI)': '–', 'P heterogeneity': '–'})
Bt = pd.DataFrame(rows)
Bt.to_csv(OUT + 'ST3b_sensitivity_missing_block.csv', index=False)

NOTE_A = ('Age gaps are residuals from linear regression of proteomic age on chronological age in all 52,995 '
          'participants with baseline proteomic data, expressed in years before standardisation within the analytic '
          'cohort (n = 40,674). Variance explained: R² of a linear model of the age gap (years) on age and age² in the '
          'analytic cohort. >50% missing: 5,197 participants of the analytic cohort in whom more than half of the 2,923 '
          'protein values were missing (median-filled) in the supplied Olink matrix. [authors]: values from the '
          'clock-training records, to be completed by the authors. MAE, mean absolute error.')
NOTE_B = ('Cox models with follow-up time as the time scale, adjusted for age, age², sex, ethnicity, Townsend index, '
          'education, smoking, alcohol intake and CKM stage (model b). Age gaps, aged-organ flags and their '
          'stage-0 thresholds were kept as in the main analysis. >50% missing: 5,197 participants (12.8%) in whom more than '
          'half of the 2,923 protein values were missing (median-filled); ≥40% missing: 6,282 participants (15.4%), which '
          'removes the whole cluster of participants with missing values for a block of about 1,460 proteins. '
          'P heterogeneity: Wald test of the product term between the age gap and an indicator of >50% missing values in '
          'the full cohort. Participants with the outcome at baseline were excluded from that outcome, as in the main analysis.')
with pd.ExcelWriter(OUT + 'Supplementary_Table_3_revised.xlsx') as w:
    A.to_excel(w, 'ST3a', index=False); Bt.to_excel(w, 'ST3b', index=False)
    pd.DataFrame({'part': ['a', 'b'], 'footnote': [NOTE_A, NOTE_B]}).to_excel(w, 'footnotes', index=False)
json.dump({'note_a': NOTE_A, 'note_b': NOTE_B}, open(OUT + 'ST3_footnotes.json', 'w'), indent=1)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 20)
print(A.drop(columns=['Candidate proteins, n', 'Selected proteins, n', 'Hold-out r with age', 'Hold-out MAE, years']).to_string(index=False))
print(Bt.to_string(index=False))
