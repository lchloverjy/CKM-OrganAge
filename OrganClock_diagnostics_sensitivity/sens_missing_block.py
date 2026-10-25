"""Sensitivity analysis: exclude participants with >50% of Olink protein values missing
(filled with the cohort median, i.e. exact 0 after rank-based inverse-normal transformation,
in the supplied matrix ukb_protein_wide_ins0_final.rds).

Wang et al. (Nat Aging 2025/2026) excluded participants with >50% missing proteins before
training the organ clocks. This script asks whether the main associations change when these
participants are removed from the analytic cohort. Age gaps, aged-organ flags and covariates
are left exactly as in the main analysis (same standardisation, same stage-0 thresholds).

Outputs (results/):
  S_missing_tier1.csv   - paired clock HR per s.d. (model b: COV_A + stage), full vs restricted,
                          plus HR within the high-missing group and P for heterogeneity
  S_missing_burden.csv  - HR for 1 and >=2 vs 0 aged organs (COV_A + stage), full vs restricted
  S_missing_meta.json   - counts
"""
import sys, os, json
sys.path.insert(0, '/home/ubuntu/work/v5/run/analysis')
os.chdir('/home/ubuntu/work/v5/run')
import numpy as np, pandas as pd
from v3_common import subcohort, cox_hr, COV_A, STG, TIER1

OUT = '/home/ubuntu/work/v5/methods_clock/results/'
os.makedirs(OUT, exist_ok=True)
COVB = COV_A + STG

s = subcohort()
z = pd.read_parquet('/home/ubuntu/work/v5/protein_zero_share.parquet')[['eid', 'zshare']]
s = s.merge(z, on='eid', how='left')
assert s.zshare.notna().all()
s['hi'] = (s.zshare > 0.5).astype(float)
s['b1'] = (s.bur_c == 1).astype(float)
s['b2'] = (s.bur_c == 2).astype(float)
R = s[s.hi == 0].copy()
H = s[s.hi == 1].copy()
# missingness is clustered just below and above 50% (whole-panel failures), so a second
# definition excludes the whole cluster (>=40% missing)
R40 = s[s.zshare < 0.4].copy()
meta = {'n_cohort': int(len(s)), 'n_high_missing': int(len(H)), 'n_restricted': int(len(R)),
        'pct_high_missing': round(100 * len(H) / len(s), 2),
        'n_ge40_missing': int((s.zshare >= 0.4).sum()), 'n_restricted_40': int(len(R40)),
        'zshare_median_pct_all': round(100 * float(s.zshare.median()), 2),
        'stage_dist_high': H.stage_use.value_counts(normalize=True).sort_index().round(4).to_dict()
        if 'stage_use' in H else None}

rows = []
for k, o in TIER1:
    g = 'gap_' + o
    for lab, d in [('Full cohort', s), ('Excluding >50% missing', R), ('Excluding >=40% missing', R40),
                   ('>50% missing only', H)]:
        r, _ = cox_hr(d, f't_{k}', f'ev_{k}', [g], COVB)
        r = r.iloc[0].to_dict(); r.update(outcome=k, clock=o, sample=lab); rows.append(r)
    # heterogeneity: gap x hi interaction in the full cohort
    d = s.copy(); d['gxh'] = d[g] * d.hi
    r, _ = cox_hr(d, f't_{k}', f'ev_{k}', [g, 'gxh', 'hi'], COVB)
    rows.append({'outcome': k, 'clock': o, 'sample': 'P heterogeneity', 'p': float(r.iloc[1].p)})
T1 = pd.DataFrame(rows)
T1.to_csv(OUT + 'S_missing_tier1.csv', index=False)

rows = []
for k in ['CKMC', 'HF', 'ASCVD', 'CKD', 'T2D', 'CLD', 'DEM']:
    for lab, d in [('Full cohort', s), ('Excluding >50% missing', R), ('Excluding >=40% missing', R40)]:
        r, _ = cox_hr(d, f't_{k}', f'ev_{k}', ['b1', 'b2'], COVB)
        for _, x in r.iterrows():
            x = x.to_dict(); x.update(outcome=k, sample=lab); rows.append(x)
B = pd.DataFrame(rows)
B.to_csv(OUT + 'S_missing_burden.csv', index=False)

# ratio of HRs (restricted / full) summary
f = T1[T1['sample'] == 'Full cohort'].set_index('outcome')
e = T1[T1['sample'] == 'Excluding >50% missing'].set_index('outcome')
meta['tier1_hr_full'] = {k: [round(f.loc[k, 'hr'], 3), round(f.loc[k, 'lo'], 3), round(f.loc[k, 'hi'], 3), int(f.loc[k, 'events'])] for k in f.index}
meta['tier1_hr_restricted'] = {k: [round(e.loc[k, 'hr'], 3), round(e.loc[k, 'lo'], 3), round(e.loc[k, 'hi'], 3), int(e.loc[k, 'events'])] for k in e.index}
e4 = T1[T1['sample'] == 'Excluding >=40% missing'].set_index('outcome')
meta['tier1_hr_restricted_40'] = {k: [round(e4.loc[k, 'hr'], 3), round(e4.loc[k, 'lo'], 3), round(e4.loc[k, 'hi'], 3), int(e4.loc[k, 'events'])] for k in e4.index}
meta['tier1_p_het'] = {r.outcome: float(r.p) for r in T1[T1['sample'] == 'P heterogeneity'].itertuples()}
bf = B[(B['sample'] == 'Full cohort') & (B.term == 'b2')].set_index('outcome')
be = B[(B['sample'] == 'Excluding >50% missing') & (B.term == 'b2')].set_index('outcome')
meta['b2_hr_full'] = {k: [round(bf.loc[k, 'hr'], 3), round(bf.loc[k, 'lo'], 3), round(bf.loc[k, 'hi'], 3)] for k in bf.index}
meta['b2_hr_restricted'] = {k: [round(be.loc[k, 'hr'], 3), round(be.loc[k, 'lo'], 3), round(be.loc[k, 'hi'], 3)] for k in be.index}
b4 = B[(B['sample'] == 'Excluding >=40% missing') & (B.term == 'b2')].set_index('outcome')
meta['b2_hr_restricted_40'] = {k: [round(b4.loc[k, 'hr'], 3), round(b4.loc[k, 'lo'], 3), round(b4.loc[k, 'hi'], 3)] for k in b4.index}
json.dump(meta, open(OUT + 'S_missing_meta.json', 'w'), indent=1, default=str)
print(json.dumps(meta, indent=1, default=str))
