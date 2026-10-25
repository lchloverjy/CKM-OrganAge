"""Figure 1 (v2, graphical-abstract overview) analyses.

Outputs (results/v5_fig1v2/):
  F1_facts.json              cohort, follow-up and imaging facts
  F1b_clocks.csv             1 s.d. of each age gap in years; target-organ flag; r(age gap, age)
  F1c_stage_profile.csv      baseline-verified stage: n, %, age, women, PREVENT-CVD
  F1d_corr_pearson.csv       Pearson correlation matrix (clustered order)
  F1d_corr_spearman.csv      Spearman correlation matrix (same order)
  F1d_corr_stage0.csv        Pearson correlation matrix in stage 0
  F1d_linkage.csv            average-linkage merge table (1 - r distance)
  F1d_pca.csv                eigenvalues, % variance, cumulative %, loadings of PC1
  F1d_independence.csv       PC1 share, PCs for 80%, Meff (Li-Ji, Nyholt), median r; bootstrap CIs
  ST1a_selection.csv         stepwise participant selection
  ST1b_analytic_samples.csv  analytic sample and events for every analysis
  ST1c_included_excluded.csv baseline characteristics of included and excluded participants (SMD)
  ST1d_staging_missingness.csv  missing staging components among participants without staging
"""
import sys, os, json
sys.path.insert(0, 'analysis')
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from v3_common import *

R2 = 'results/v5_fig1v2/'
os.makedirs(R2, exist_ok=True)
rng = np.random.default_rng(20261018)

s = subcohort()
N = len(s)
facts = {'n_cohort': N}

# ------------------------------------------------------------------ follow-up / imaging facts
fu = s.fu_years
facts.update(fu_median=float(fu.median()), fu_q1=float(fu.quantile(.25)), fu_q3=float(fu.quantile(.75)),
             person_years=float(fu.sum()), deaths=int(s.ev_DEATH.sum()),
             n_imaging_any=int(((s.has_cmr.fillna(0) + s.has_brain.fillna(0)) > 0).sum()),
             n_cmr=int(s.has_cmr.sum()), n_brain=int(s.has_brain.sum()),
             n_asi=int(s.asi.notna().sum()), n_aortic=int(s.asc_dist.notna().sum()))
facts['n_reclassified'] = int((s.stage_bv != s.stage_flag).sum())

# ------------------------------------------------------------------ b: clocks
rows = []
for o in ORG:
    rows.append(dict(clock=o, sd_years=float(s['yr_' + o].std()), target=int(o in TARGET),
                     whole_body=int(o in ('Organismal', 'Conventional')),
                     r_with_age=float(np.corrcoef(s['yr_' + o], s.age)[0, 1]),
                     mean_years=float(s['yr_' + o].mean())))
pd.DataFrame(rows).to_csv(R2 + 'F1b_clocks.csv', index=False)

# ------------------------------------------------------------------ c: stage profile
rows = []
for k in range(4):
    g = s[s.stage_bv == k]
    rows.append(dict(stage=k, n=len(g), pct=100 * len(g) / N, age_mean=g.age.mean(), age_sd=g.age.std(),
                     women_pct=100 * (1 - g.male).mean(),
                     prevent_median=g.prevent_bv.median(), prevent_q1=g.prevent_bv.quantile(.25),
                     prevent_q3=g.prevent_bv.quantile(.75),
                     prevent_ge75_pct=100 * (g.prevent_bv >= 7.5).mean(),
                     egfr_median=g.egfr_cc.median(),
                     n_provider=int((s.stage_flag == k).sum())))
pd.DataFrame(rows).to_csv(R2 + 'F1c_stage_profile.csv', index=False)

# ------------------------------------------------------------------ d: independence of the clocks
X = s[['gap_' + o for o in ORG]].to_numpy(float)
Cp = np.corrcoef(X, rowvar=False)
Dm = 1 - Cp
np.fill_diagonal(Dm, 0)
L = optimal_leaf_ordering(linkage(squareform(Dm, checks=False), method='average'), squareform(Dm, checks=False))
order = [ORG[i] for i in leaves_list(L)]
idx = [ORG.index(o) for o in order]


def mat(C, name):
    pd.DataFrame(C[np.ix_(idx, idx)], index=order, columns=order).to_csv(R2 + name)


mat(Cp, 'F1d_corr_pearson.csv')
Cs = pd.DataFrame(X).rank().corr().to_numpy()
mat(Cs, 'F1d_corr_spearman.csv')
X0 = s.loc[s.stage_bv == 0, ['gap_' + o for o in ORG]].to_numpy(float)
C0 = np.corrcoef(X0, rowvar=False)
mat(C0, 'F1d_corr_stage0.csv')
pd.DataFrame(L, columns=['i', 'j', 'dist', 'n']).assign(
    labels=';'.join(ORG), leaf_order=';'.join(order)).to_csv(R2 + 'F1d_linkage.csv', index=False)


def indep(C):
    M = C.shape[0]
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    share = lam / lam.sum()
    cum = np.cumsum(share)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    nyholt = 1 + (M - 1) * (1 - np.var(lam, ddof=1) / M)
    a = np.abs(lam)
    liji = float(np.sum((a >= 1).astype(float) + (a - np.floor(a))))
    off = C[np.triu_indices(M, 1)]
    return dict(pc1_pct=100 * share[0], pc2_pct=100 * share[1], k80=k80, meff_liji=liji, meff_nyholt=nyholt,
                r_median=float(np.median(off)), r_min=float(off.min()), r_max=float(off.max()),
                r_mean=float(off.mean())), lam


res, lam = indep(Cp)
res0, lam0 = indep(C0)
resS, _ = indep(Cs)
# bootstrap (200 resamples) for the full-cohort and stage-0 summaries
B = 200
bt, bt0 = [], []
n0 = len(X0)
for b in range(B):
    ii = rng.integers(0, N, N)
    bt.append(indep(np.corrcoef(X[ii], rowvar=False))[0])
    jj = rng.integers(0, n0, n0)
    bt0.append(indep(np.corrcoef(X0[jj], rowvar=False))[0])
bt, bt0 = pd.DataFrame(bt), pd.DataFrame(bt0)
out = []
for lab, r, bb in [('All (n = %d)' % N, res, bt), ('Stage 0 (n = %d)' % n0, res0, bt0), ('All, Spearman', resS, None)]:
    for k, v in r.items():
        lo = hi = np.nan
        if bb is not None:
            lo, hi = bb[k].quantile([.025, .975])
        out.append(dict(sample=lab, metric=k, estimate=v, lo=lo, hi=hi))
pd.DataFrame(out).to_csv(R2 + 'F1d_independence.csv', index=False)
# difference in median r, all vs stage 0
dmed = bt.r_median.values - bt0.r_median.values
facts['r_median_diff_all_minus_stage0'] = [float(res['r_median'] - res0['r_median']),
                                           float(np.quantile(dmed, .025)), float(np.quantile(dmed, .975))]
share = lam / lam.sum()
vec = np.linalg.eigh(Cp)[1][:, ::-1][:, 0]
vec = vec * np.sign(vec.sum())
pd.DataFrame(dict(pc=np.arange(1, 13), eigenvalue=lam, pct=100 * share, cum_pct=100 * np.cumsum(share),
                  eigenvalue_stage0=lam0, pct_stage0=100 * lam0 / lam0.sum(),
                  pc1_loading_clock=ORG, pc1_loading=vec)).to_csv(R2 + 'F1d_pca.csv', index=False)

# ------------------------------------------------------------------ Supplementary Table 1
U = pd.read_parquet(D + 'data_all.parquet').rename(columns={'Eid': 'eid'})
G = pd.read_parquet(D + 'age_gap.parquet')
A = pd.read_parquet(D + 'analysis.parquet', columns=['eid', 'stage', 'has_gap'])
n_ukb = len(U)
n_gap = len(G)
staged = A[A.has_gap == 1]
n_staged = len(staged)
n_st4 = int((staged.stage == 4).sum())
rows = [
    dict(step=1, description='UK Biobank participants recruited 2006-2010', n_remaining=n_ukb, n_excluded=np.nan,
         reason=''),
    dict(step=2, description='Baseline plasma proteomics (UK Biobank Pharma Proteomics Project, Olink Explore 3072; 2,923 proteins) '
         'with all 12 proteomic organ age gaps', n_remaining=n_gap, n_excluded=n_ukb - n_gap,
         reason='No baseline proteomic data; the protein and organ age gap files match one to one (52,995 participants)'),
    dict(step=3, description='Data required for CKM staging available', n_remaining=n_staged,
         n_excluded=n_gap - n_staged, reason='Missing data required for CKM staging'),
    dict(step=4, description='Analytic cohort: CKM stages 0-3', n_remaining=n_staged - n_st4, n_excluded=n_st4,
         reason='CKM stage 4 (clinical cardiovascular disease) at baseline'),
    dict(step=5, description='Baseline-verified CKM stage (no exclusion)', n_remaining=N, n_excluded=0,
         reason=f'{facts["n_reclassified"]} participants ({100 * facts["n_reclassified"] / N:.1f}%) reclassified'),
    dict(step=6, description='Linked hospital inpatient and death records to 30 September 2021 (no exclusion)',
         n_remaining=N, n_excluded=0, reason='Follow-up censored at death or end of follow-up'),
]
assert n_staged - n_st4 == N
pd.DataFrame(rows).to_csv(R2 + 'ST1a_selection.csv', index=False)

# characteristics of included and excluded participants
grp = pd.Series('Excluded: no CKM staging data', index=G.eid.values)
grp[staged.eid[staged.stage == 4].values] = 'Excluded: CKM stage 4'
grp[s.eid.values] = 'Included: analytic cohort'
H = G.assign(group=grp.loc[G.eid].values).merge(U, on='eid', how='left')
H['women'] = (H.Sex == 0).astype(float)
H['nonwhite'] = np.where(H.Ethnicity.isin(['White']), 0.0, np.where(H.Ethnicity.isin([None, '-1']), np.nan, 1.0))
H.loc[H.Ethnicity.isna(), 'nonwhite'] = np.nan
H['current_smoker'] = np.where(H.Smoking_status.isin([0, 1, 2]), (H.Smoking_status == 2).astype(float), np.nan)
H['degree'] = np.where(H.Qualifications.isin([-3]) | H.Qualifications.isna(), np.nan, (H.Qualifications == 1).astype(float))
H['hba1c_ge48'] = np.where(H.HbA1c.notna(), (H.HbA1c >= 48).astype(float), np.nan)
H['org_gap'] = H['Organismal_age_gap_boruta']
H['conv_gap'] = H['非特异Organismal_age_gap_boruta']
H['kid_gap'] = H['Kidney_age_gap_boruta']
H['heart_gap'] = H['Heart_age_gap_boruta']
VARS = [('Age, years', 'Age', 'mean'), ('Women, %', 'women', 'pct'), ('Ethnicity other than White, %', 'nonwhite', 'pct'),
        ('Townsend deprivation index', 'townsend_index', 'mean'), ('College or university degree, %', 'degree', 'pct'),
        ('Current smoker, %', 'current_smoker', 'pct'), ('Body mass index, kg/m2', 'BMI', 'mean'),
        ('Systolic blood pressure, mmHg', 'SBP', 'mean'), ('HbA1c, mmol/mol', 'HbA1c', 'mean'),
        ('HbA1c >= 48 mmol/mol, %', 'hba1c_ge48', 'pct'), ('LDL cholesterol, mmol/L', 'LDLc', 'mean'),
        ('HDL cholesterol, mmol/L', 'HDLc', 'mean'), ('Triglycerides, mmol/L', 'TG', 'median'),
        ('C-reactive protein, mg/L', 'CRP', 'median'), ('Creatinine, umol/L', 'Creatinine', 'mean'),
        ('Cystatin C, mg/L', 'CysC', 'mean'), ('Organismal age gap, years', 'org_gap', 'mean'),
        ('Conventional age gap, years', 'conv_gap', 'mean'), ('Kidney age gap, years', 'kid_gap', 'mean'),
        ('Heart age gap, years', 'heart_gap', 'mean')]
GR = ['Included: analytic cohort', 'Excluded: no CKM staging data', 'Excluded: CKM stage 4']
rows = []
inc = H[H.group == GR[0]]
for lab, v, kind in VARS:
    r = dict(characteristic=lab, kind=kind)
    for gname in GR:
        x = H.loc[H.group == gname, v].astype(float)
        r[gname + ' | n'] = int((H.group == gname).sum())
        r[gname + ' | missing %'] = 100 * x.isna().mean()
        x = x.dropna()
        if kind == 'pct':
            r[gname + ' | value'] = f'{100 * x.mean():.1f}'
        elif kind == 'median':
            r[gname + ' | value'] = f'{x.median():.2f} ({x.quantile(.25):.2f}-{x.quantile(.75):.2f})'
        else:
            r[gname + ' | value'] = f'{x.mean():.2f} ({x.std():.2f})'
        if gname != GR[0]:
            xi = inc[v].astype(float).dropna()
            if kind == 'pct':
                p1, p2 = x.mean(), xi.mean()
                smd = (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
            else:
                xx, yy = (np.log(x.clip(lower=0.01)), np.log(xi.clip(lower=0.01))) if kind == 'median' else (x, xi)
                smd = (xx.mean() - yy.mean()) / np.sqrt((xx.var() + yy.var()) / 2)
            r[gname + ' | SMD vs included'] = smd
    rows.append(r)
pd.DataFrame(rows).to_csv(R2 + 'ST1c_included_excluded.csv', index=False)

# missingness of staging components among participants without staging
COMP = [('BMI', 'BMI'), ('Systolic blood pressure', 'SBP'), ('HbA1c', 'HbA1c'), ('Fasting/random glucose', 'Glucose'),
        ('Triglycerides', 'TG'), ('HDL cholesterol', 'HDLc'), ('Total cholesterol', 'CHOL'), ('Creatinine', 'Creatinine'),
        ('Cystatin C', 'CysC'), ('Urine creatinine (UACR)', 'Urine_creatinine')]
rows = []
for lab, v in COMP:
    r = dict(component=lab)
    for gname in GR:
        r[gname] = 100 * H.loc[H.group == gname, v].isna().mean()
    rows.append(r)
nost = H[H.group == GR[1]]
anymiss = nost[[v for _, v in COMP if v not in ('Glucose', 'Urine_creatinine')]].isna().any(axis=1).mean() * 100
facts['nostage_any_core_component_missing_pct'] = float(anymiss)
pd.DataFrame(rows).to_csv(R2 + 'ST1d_staging_missingness.csv', index=False)

# ------------------------------------------------------------------ analytic samples (ST1b)
t12 = pd.read_csv('/home/ubuntu/work/v5/deliver/v5_submission/03_Main_Figures/Figure1/source_data/Fig1e_tier12_events.csv')
t3 = pd.read_csv('/home/ubuntu/work/v5/deliver/v5_submission/03_Main_Figures/Figure1/source_data/Fig1e_tier3_events.csv')
F6 = pd.read_csv('/home/ubuntu/work/v5/figure6/source_data/Fig6_facts.csv')
f6 = dict(zip(F6.iloc[:, 0], F6.iloc[:, 1]))
facts['fig6_facts'] = {k: str(v) for k, v in f6.items()}
st12 = int(s.stage_bv.isin([1, 2]).sum())
_im = pd.read_csv('results/v5_fig6/F6f_imaging.csv')
IMN = {'Heart': int(_im[_im.domain == 'Heart'].n.max()), 'Brain': int(_im[_im.domain == 'Brain'].n.max()),
       'aorta': int(_im[_im['var'].isin(['asc_dist', 'desc_dist'])].n.max()), 'asi': int(_im[_im['var'] == 'asi'].n.max())}
rows = [
    ('Fig. 2a-e', 'Organ age gaps across CKM stages (cross-sectional)', 'Analytic cohort', N, ''),
    ('Fig. 2f,g', 'Multi-organ ageing burden; identification of high burden', 'Stages 1-2', st12, ''),
]
for _, r in t12[t12.tier1 == 1].iterrows():
    rows.append(('Fig. 3b', f'Tier 1: {r.disease}', 'Free of the outcome at baseline', int(r.n_at_risk), int(r.n_events)))
rows += [
    ('Fig. 3a,c,e', 'Tier 2: 46 clinically defined diseases', 'Free of each outcome at baseline (range)',
     f'{int(t12.n_at_risk.min()):,}-{int(t12.n_at_risk.max()):,}', f'{int(t12.n_events.min()):,}-{int(t12.n_events.max()):,}'),
    ('Fig. 3d', 'Tier 3: 305 phecodes with >= 80 events', 'Free of each phecode at baseline (range)',
     f'{int(t3.n_at_risk.min()):,}-{int(t3.n_at_risk.max()):,}', f'{int(t3.n_events.min()):,}-{int(t3.n_events.max()):,}'),
    ('Fig. 4a-d', 'CKM composite outcome by stage and burden', 'Free of all components at baseline', 37904, 6652),
    ('Fig. 4e', 'Multisystem disease trajectories', 'Analytic cohort', N, '22,047 first / 12,594 second system'),
    ('Fig. 5', 'Incremental prediction over model G and model D', 'Analytic cohort (composite: 37,904)', N, 'per outcome'),
    ('Fig. 5d', 'Case capture of the top 10% of predicted risk', 'Stages 1-2', st12, 'per outcome'),
    ('Fig. 6', 'PREVENT calibration and risk enhancement (total CVD)', 'Free of CVD at baseline', 40168, 4207),
    ('Fig. 6', 'Comparison with guideline risk-enhancing factors (complete case)', 'All enhancers measured', 32158, 3303),
    ('Fig. 6', 'Testing strategies', 'Stages 1-2', st12, ''),
    ('Fig. 6', 'Imaging: cardiac MRI', 'Imaging visit; analysed (measured)', IMN['Heart'], f"measured {facts['n_cmr']:,}"),
    ('Fig. 6', 'Imaging: aortic distensibility (MRI)', 'Imaging visit; analysed (measured)', IMN['aorta'], f"measured {facts['n_aortic']:,}"),
    ('Fig. 6', 'Imaging: brain MRI', 'Imaging visit; analysed (measured)', IMN['Brain'], f"measured {facts['n_brain']:,}"),
    ('Fig. 6', 'Arterial stiffness index (finger photoplethysmography)', 'Baseline; analysed (measured)', IMN['asi'], f"measured {facts['n_asi']:,}"),
    ('Supplementary Note 4', 'Lifestyle analyses', 'Complete data on seven lifestyle factors', 33613, ''),
]
pd.DataFrame(rows, columns=['figure', 'analysis', 'sample', 'n', 'events_or_note']).to_csv(R2 + 'ST1b_analytic_samples.csv', index=False)

json.dump(facts, open(R2 + 'F1_facts.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in facts.items() if k != 'fig6_facts'}, indent=1, default=float))
print(pd.read_csv(R2 + 'F1d_independence.csv').round(3).to_string())
print(pd.read_csv(R2 + 'F1c_stage_profile.csv').round(2).to_string())
print(pd.read_csv(R2 + 'F1b_clocks.csv').round(3).to_string())
print(order)
