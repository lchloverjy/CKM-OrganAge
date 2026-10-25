"""Supporting estimates for the Results text of Fig. 2 that were not stored in the figure source data:
(1) 95% CIs for the cross-validated AUCs and AUC differences of nested models identifying high burden in stages 1-2;
(2) distribution of PREVENT risk among high- and low-burden participants in stages 1-2;
(3) burden distribution and pairwise co-occurrence after residualising age gaps on kidney function.
Definitions follow 02_layer1.py, 03_layer2.py and prep_v2.py."""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from patsy import dmatrix
from sklearn.metrics import roc_auc_score
sys.path.insert(0, './code')
from importlib import import_module
from stats_utils import cv_auc
warnings.filterwarnings("ignore")
L1 = import_module('02_layer1'); L2 = import_module('03_layer2')
OUT = './v4/Supplementary/fig2_text_checks/'
import os; os.makedirs(OUT, exist_ok=True)

a = L1.load().reset_index(drop=True)
bu = pd.read_parquet('./data/burden.parquet')
a = a.merge(bu, on='eid', how='left')
res = {}

# ---------------- (1) AUC with bootstrap CI in stages 1-2
b = a[a.stage.isin([1, 2])].copy().reset_index(drop=True)
y = (b.burden_t >= 2).astype(int).values
b['log_prevent'] = np.log(b.prevent10)
feats = {'Age, sex': ['age', 'age2', 'male'],
         '+ CKM stage': ['age', 'age2', 'male', 'st2'],
         '+ PREVENT risk': ['age', 'age2', 'male', 'st2', 'log_prevent'],
         '+ routine clinical panel': ['age', 'age2', 'male', 'st2', 'log_prevent', 'bmi', 'waist', 'hba1c', 'sbp', 'dbp',
                                      'log_tg_hdl', 'ldl', 'hdl', 'egfr', 'log_crp', 'log_nlr', 'htn_med', 'statin',
                                      'diabetes', 'smoke1', 'smoke2', 'townsend', 'degree'],
         'Routine clinical panel without eGFR': ['age', 'age2', 'male', 'st2', 'log_prevent', 'bmi', 'waist', 'hba1c', 'sbp', 'dbp',
                                      'log_tg_hdl', 'ldl', 'hdl', 'log_crp', 'log_nlr', 'htn_med', 'statin',
                                      'diabetes', 'smoke1', 'smoke2', 'townsend', 'degree']}
P = {}
for k, c in feats.items():
    p, _ = cv_auc(b[c].values, y, n_repeats=5)
    P[k] = p
rng = np.random.default_rng(11)
B = 1000
boots = {k: [] for k in P}
idx_all = np.arange(len(y))
for _ in range(B):
    i = rng.choice(idx_all, len(y), replace=True)
    for k in P:
        boots[k].append(roc_auc_score(y[i], P[k][i]))
rows = []
order = list(P)
for j, k in enumerate(order):
    auc = roc_auc_score(y, P[k]); bb = np.array(boots[k])
    r = dict(model=k, auc=auc, lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5))
    if 0 < j < 4:
        prev = order[j - 1]; d = bb - np.array(boots[prev])
        r.update(delta_vs_previous=auc - roc_auc_score(y, P[prev]), d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5))
    if j == 4:
        full = '+ routine clinical panel'; d = np.array(boots[full]) - bb
        r.update(delta_full_minus_this=roc_auc_score(y, P[full]) - auc, d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5))
    rows.append(r)
AUC = pd.DataFrame(rows); AUC.to_csv(OUT + 'auc_ci.csv', index=False); print(AUC.round(3).to_string())
res['n_stage12'] = int(len(y)); res['n_high_stage12'] = int(y.sum())

# ---------------- (2) PREVENT among high and low burden (stages 1-2)
pr = b.prevent10 * (100 if b.prevent10.max() <= 1 else 1)
hi = y == 1
res['prevent_lt5_high_pct'] = float((pr[hi] < 5).mean() * 100)
res['prevent_lt5_low_pct'] = float((pr[~hi] < 5).mean() * 100)
res['prevent_lt75_high_pct'] = float((pr[hi] < 7.5).mean() * 100)
res['prevent_median_high'] = float(pr[hi].median()); res['prevent_median_low'] = float(pr[~hi].median())
res['n_high_prevent_lt5'] = int((pr[hi] < 5).sum())
for s in [1, 2]:
    m = (b.stage == s).values
    res[f'stage{s}_n_high'] = int((hi & m).sum())
    res[f'stage{s}_prevent_lt5_high_pct'] = float((pr[hi & m] < 5).mean() * 100)

# ---------------- (3) kidney-function residualised age gaps
a['cysc_f'] = a.cysc.fillna(a.cysc.median())
# natural cubic regression splines (4 df) of eGFR and cystatin C; reproduces results/L2_burden_dist_egfr_resid.csv exactly
Z = dmatrix('cr(egfr, df=4) + cr(cysc_f, df=4)', a, return_type='dataframe')
ar = a.copy()
for o in L2.ORG10:
    g = a['gap_' + o]
    ar['gap_' + o] = g - sm.OLS(g, Z).fit().fittedvalues
F = L2.extreme_flags(ar)
bt = F[L2.TARGET].sum(1)
dist = pd.crosstab(a.stage, bt.clip(upper=3), normalize='index')
dist.to_csv(OUT + 'resid_burden_dist.csv'); print(dist.round(4))
if not os.path.exists('./results/L2_burden_dist_egfr_resid.csv'):   # v5: regenerate the stored copy on the corrected cohort
    dist.to_csv('./results/L2_burden_dist_egfr_resid.csv')
ref = pd.read_csv('./results/L2_burden_dist_egfr_resid.csv', index_col=0)
res['resid_max_abs_diff_vs_stored'] = float(np.abs(dist.values - ref.values).max())
res['resid_n_stage12_ge2'] = int(((a.stage.isin([1, 2])) & (bt >= 2)).sum())
res['resid_ge2_by_stage_pct'] = {int(k): float(v * 100) for k, v in (bt >= 2).groupby(a.stage).mean().items()}
# co-occurrence with within-stage permutation, as in 03_layer2.py
rng = np.random.default_rng(5)
S = a.stage.values; Fm = F[L2.TARGET].values
obs = Fm[S >= 1].T @ Fm[S >= 1]
perm = []
for _ in range(500):
    Pm = Fm.copy()
    for k in [0, 1, 2, 3]:
        ii = np.where(S == k)[0]
        for j in range(Pm.shape[1]):
            Pm[ii, j] = Pm[rng.permutation(ii), j]
    perm.append(Pm[S >= 1].T @ Pm[S >= 1])
perm = np.array(perm); exp = perm.mean(0); pv = ((perm >= obs).sum(0) + 1) / (len(perm) + 1)
co = [dict(o1=oi, o2=oj, obs=int(obs[i, j]), exp=exp[i, j], ratio=obs[i, j] / exp[i, j], p=pv[i, j])
      for i, oi in enumerate(L2.TARGET) for j, oj in enumerate(L2.TARGET) if j > i]
CO = pd.DataFrame(co); CO.to_csv(OUT + 'resid_cooccurrence.csv', index=False); print(CO.round(3).to_string())
res['resid_cooc_ratio_range'] = [float(CO.ratio.min()), float(CO.ratio.max())]
res['resid_cooc_max_p'] = float(CO.p.max())
json.dump(res, open(OUT + 'summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

# ---------------- (4) counts by stage for the main definition
cnt = pd.DataFrame({'n': a.groupby('stage').size(),
                    'n_ge2': (a.burden_t >= 2).groupby(a.stage).sum(),
                    'n_ge3': (a.burden_t >= 3).groupby(a.stage).sum(),
                    'n_zero': (a.burden_t == 0).groupby(a.stage).sum()})
cnt.to_csv(OUT + 'burden_counts.csv'); print(cnt)
