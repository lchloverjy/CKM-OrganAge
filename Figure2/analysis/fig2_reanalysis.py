"""Additional analyses for Figure 2 (cross-sectional cohort, n = 40,674).

(1) Onset share of the stage 3 difference (beta1/beta3, beta2/beta3) for the 12 clocks under M2, with 200 joint bootstrap
    samples, a bootstrap Wald test of equal onset shares and Ward clustering of the stage profiles with bootstrap reproducibility.
(2) Gelbach decomposition of the change in each stage coefficient from M2 to M3 into the contributions of the six stage-defining
    traits (grouped into the five CKM staging domains), with 200 bootstrap samples.
(3) Within-stage heterogeneity: variance in each age gap explained by stage beyond the M2 covariates, the percentage of stage 1-2
    participants whose M2-adjusted age gap exceeds the stage 3 median, and the probability of superiority of stage 3 over stage 0.
(4) Stage differences (M2) after residualising age gaps on natural cubic splines of eGFR and cystatin C.
(5) PREVENT risk categories among stage 1-2 participants with high and low organ ageing burden.
Model definitions follow 02_layer1.py (M1-M3) and 03_layer2.py (extreme ageing and burden). Outputs: results/v4/F2_*.csv/json.
"""
import sys, json, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import chi2, rankdata, gaussian_kde
from scipy.cluster.hierarchy import linkage, fcluster
warnings.filterwarnings('ignore')
sys.path.insert(0, './code')
from importlib import import_module
from stats_utils import ols_terms
L1 = import_module('02_layer1')

OUT = './results/v4/'
ORG = L1.ORG
ST = ['st1', 'st2', 'st3']
C2 = L1.COV2
DEF = L1.DEF                                   # bmi, waist, hba1c, sbp, log_tg_hdl, egfr
DOMAIN = {'bmi': 'Adiposity', 'waist': 'Adiposity', 'hba1c': 'Glycaemia', 'sbp': 'Blood pressure',
          'log_tg_hdl': 'Lipids', 'egfr': 'Kidney function'}
DOMS = ['Adiposity', 'Glycaemia', 'Blood pressure', 'Lipids', 'Kidney function']
LATE_SET = None
NB = 200

a = L1.load().reset_index(drop=True)
bu = pd.read_parquet('./data/burden.parquet')
a = a.merge(bu, on='eid', how='left')
n = len(a)
SD = np.array([a['gap_' + o].std() for o in ORG])
Y = np.column_stack([a['gap_' + o].values for o in ORG]) / SD      # standardised age gaps (s.d. of the full cohort)
X2 = sm.add_constant(a[ST + C2]).values
X3 = sm.add_constant(a[ST + C2 + DEF]).values
X0 = sm.add_constant(a[C2]).values                                 # M2 covariates without stage
D = a[DEF].values
stage = a.stage.values
a['cysc_f'] = a.cysc.fillna(a.cysc.median())
ZK = dmatrix('cr(egfr, df=4) + cr(cysc_f, df=4)', a, return_type='dataframe').values


def lsq(X, Yv):
    return np.linalg.lstsq(X, Yv, rcond=None)[0]


def auc_ps(score, lab):
    """Probability that a random case (lab = 1) has a higher score than a random control (Mann-Whitney)."""
    r = rankdata(score)
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def estimates(idx):
    """All Figure 2 quantities for one (bootstrap) sample of row indices."""
    x2, x3, x0, y, d, s, zk = X2[idx], X3[idx], X0[idx], Y[idx], D[idx], stage[idx], ZK[idx]
    b2 = lsq(x2, y)                       # rows: const, st1, st2, st3, covariates
    b3 = lsq(x3, y)
    g = lsq(x2, d)[1:4]                   # conditional stage differences in each trait (3 x 6)
    lam = b3[-len(DEF):]                  # trait coefficients in M3 (6 x 12)
    dec = g[:, :, None] * lam[None, :, :]  # stage x trait x clock
    # within-stage heterogeneity on M2-adjusted gaps (stage omitted)
    r0 = y - x0 @ lsq(x0, y)
    rss0 = (r0 ** 2).sum(0)
    r2 = y - x2 @ b2
    tss = ((y - y.mean(0)) ** 2).sum(0)
    dr2 = (rss0 - (r2 ** 2).sum(0)) / tss
    m12 = np.isin(s, [1, 2]); m0 = s == 0; m3 = s == 3; m30 = m0 | m3
    med3 = np.median(r0[m3], axis=0)
    above12 = (r0[m12] > med3).mean(0); above0 = (r0[m0] > med3).mean(0)
    lab = m3[m30].astype(int)
    ps = np.array([auc_ps(r0[m30, k], lab) for k in range(y.shape[1])])
    # kidney-function residualised gaps, M2 stage differences
    yk = y - zk @ lsq(zk, y)
    bk = lsq(x2, yk)[1:4]
    return dict(b2=b2[1:4], b3=b3[1:4], dec=dec, dr2=dr2, above12=above12, above0=above0, ps=ps, bk=bk)


full = estimates(np.arange(n))
rng = np.random.default_rng(2024)
BS = []
for bb in range(NB):
    BS.append(estimates(rng.integers(0, n, n)))
    if (bb + 1) % 50 == 0:
        print('bootstrap', bb + 1, flush=True)


def pct(arrs, q):
    return np.percentile(np.stack(arrs), q, axis=0)


# ---------------------------------------------------------------- (1) onset shares, Wald test, clustering
def shares(b):
    return b[0] / b[2], b[1] / b[2]


f1, f2 = shares(full['b2'])
F1 = np.stack([shares(x['b2'])[0] for x in BS]); F2 = np.stack([shares(x['b2'])[1] for x in BS])
B2 = np.stack([x['b2'] for x in BS])
C = np.hstack([np.eye(len(ORG) - 1), -np.ones((len(ORG) - 1, 1))])
Sig = np.cov(F1.T)
W = float((C @ f1) @ np.linalg.solve(C @ Sig @ C.T, C @ f1)); pW = float(chi2.sf(W, len(ORG) - 1))


def clusters(v1, v2):
    lab = fcluster(linkage(np.column_stack([v1, v2]), 'ward'), 2, 'maxclust')
    # name the cluster with the lower mean onset share 'late'
    late = 1 if v1[lab == 1].mean() < v1[lab == 2].mean() else 2
    return lab == late


late0 = clusters(f1, f2)
late_names = [o for o, l in zip(ORG, late0) if l]
same = np.mean([np.array_equal(clusters(a_, b_), late0) for a_, b_ in zip(F1, F2)])
co_late = np.mean([clusters(a_, b_) for a_, b_ in zip(F1, F2)], axis=0)   # probability of assignment to the late group
# late-early difference in mean onset share
dl = f1[late0].mean() - f1[~late0].mean()
DL = np.array([x[late0].mean() - x[~late0].mean() for x in F1])
on = pd.DataFrame({'clock': ORG, 'beta1': full['b2'][0], 'beta2': full['b2'][1], 'beta3': full['b2'][2],
                   'beta3_years': full['b2'][2] * SD,
                   'share1': f1, 'share1_lo': np.percentile(F1, 2.5, 0), 'share1_hi': np.percentile(F1, 97.5, 0),
                   'share2': f2, 'share2_lo': np.percentile(F2, 2.5, 0), 'share2_hi': np.percentile(F2, 97.5, 0),
                   'group': np.where(late0, 'late', 'early'), 'p_late_group': co_late})
on.to_csv(OUT + 'F2_onset_share.csv', index=False)
# bootstrap CIs of the M2 stage differences themselves (for the matrix colour scale, CIs are the robust ones in L1)
summ = dict(n=int(n), n_boot=NB, wald_equal_share1=W, wald_df=len(ORG) - 1, wald_p=pW, late_clocks=late_names,
            cluster_reproduced=float(same), late_minus_early_share1=float(dl),
            late_minus_early_lo=float(np.percentile(DL, 2.5)), late_minus_early_hi=float(np.percentile(DL, 97.5)))

# ---------------------------------------------------------------- (2) Gelbach decomposition
DEC = np.stack([x['dec'] for x in BS])          # boot x stage x trait x clock
rows = []
for si in range(3):
    for k, o in enumerate(ORG):
        m2 = full['b2'][si, k]; m3 = full['b3'][si, k]
        m2b = B2[:, si, k]; m3b = np.stack([x['b3'] for x in BS])[:, si, k]
        base = dict(stage=si + 1, clock=o, m2=m2, m2_lo=np.percentile(m2b, 2.5), m2_hi=np.percentile(m2b, 97.5),
                    m3=m3, m3_lo=np.percentile(m3b, 2.5), m3_hi=np.percentile(m3b, 97.5),
                    explained=m2 - m3, explained_lo=np.percentile(m2b - m3b, 2.5), explained_hi=np.percentile(m2b - m3b, 97.5))
        for j, t in enumerate(DEF):
            v = full['dec'][si, j, k]; vb = DEC[:, si, j, k]
            rows.append({**base, 'level': 'trait', 'component': t, 'contribution': v,
                         'lo': np.percentile(vb, 2.5), 'hi': np.percentile(vb, 97.5)})
        for dm in DOMS:
            jj = [j for j, t in enumerate(DEF) if DOMAIN[t] == dm]
            v = full['dec'][si, jj, k].sum(); vb = DEC[:, si, jj, k].sum(1)
            rows.append({**base, 'level': 'domain', 'component': dm, 'contribution': v,
                         'lo': np.percentile(vb, 2.5), 'hi': np.percentile(vb, 97.5)})
GB = pd.DataFrame(rows)
chk = GB[GB.level == 'trait'].groupby(['stage', 'clock']).contribution.sum()
exp_ = GB[GB.level == 'trait'].groupby(['stage', 'clock']).explained.first()
summ['gelbach_additivity_max_abs_error'] = float((chk - exp_).abs().max())
GB.to_csv(OUT + 'F2_gelbach.csv', index=False)

# ---------------------------------------------------------------- (3) within-stage heterogeneity
WH = pd.DataFrame({'clock': ORG,
                   'r2_stage_pct': 100 * full['dr2'], 'r2_lo': 100 * pct([x['dr2'] for x in BS], 2.5),
                   'r2_hi': 100 * pct([x['dr2'] for x in BS], 97.5),
                   'above_st3_median_stage12_pct': 100 * full['above12'],
                   'above12_lo': 100 * pct([x['above12'] for x in BS], 2.5), 'above12_hi': 100 * pct([x['above12'] for x in BS], 97.5),
                   'above_st3_median_stage0_pct': 100 * full['above0'],
                   'above0_lo': 100 * pct([x['above0'] for x in BS], 2.5), 'above0_hi': 100 * pct([x['above0'] for x in BS], 97.5),
                   'prob_superiority_3v0': full['ps'], 'ps_lo': pct([x['ps'] for x in BS], 2.5), 'ps_hi': pct([x['ps'] for x in BS], 97.5)})
WH.to_csv(OUT + 'F2_within_stage.csv', index=False)
# densities of M2-adjusted standardised age gaps (no individual-level values are exported)
r0 = Y - X0 @ lsq(X0, Y)
grid = np.linspace(-4, 4, 321)
dens = []
for k, o in enumerate(ORG):
    for s in range(4):
        v = r0[stage == s, k]
        kd = gaussian_kde(v)
        dens.append(pd.DataFrame({'clock': o, 'stage': s, 'x': grid, 'density': kd(grid), 'median': np.median(v), 'n': len(v)}))
pd.concat(dens).to_csv(OUT + 'F2_density.csv', index=False)

# ---------------------------------------------------------------- (4) kidney-function residualised stage differences (M2)
yk = Y - ZK @ lsq(ZK, Y)
rows = []
for k, o in enumerate(ORG):
    res, _ = ols_terms(pd.Series(yk[:, k]), a[ST + C2], ST)
    for _, r in res.iterrows():
        rows.append(dict(clock=o, stage=int(r.term[-1]), beta_sd=r.beta, lo=r.lo, hi=r.hi, p=r.p))
KR = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
KR['q'] = multipletests(KR.p, method='fdr_bh')[1]
BK = np.stack([x['bk'] for x in BS])
fk1 = full['bk'][0] / full['bk'][2]
KR = KR.merge(pd.DataFrame({'clock': ORG, 'share1': fk1,
                            'share1_lo': np.percentile(BK[:, 0] / BK[:, 2], 2.5, 0),
                            'share1_hi': np.percentile(BK[:, 0] / BK[:, 2], 97.5, 0)}), on='clock')
KR.to_csv(OUT + 'F2_kidney_residualised_stage.csv', index=False)

# ---------------------------------------------------------------- (5) PREVENT categories by burden in stages 1-2
b = a[a.stage.isin([1, 2])].copy()
pr = b.prevent10 * (100 if b.prevent10.max() <= 1 else 1)
b['prev_cat'] = pd.cut(pr, [-np.inf, 5, 7.5, 20, np.inf], right=False, labels=['<5%', '5–<7.5%', '7.5–<20%', '≥20%'])
b['high'] = np.where(b.burden_t >= 2, 'High burden (≥2)', 'Low burden (<2)')
pc = pd.crosstab(b.high, b.prev_cat)
pcp = pd.crosstab(b.high, b.prev_cat, normalize='index') * 100
PC = pc.stack().rename('n').reset_index().merge(pcp.stack().rename('pct').reset_index(), on=['high', 'prev_cat'])
PC.to_csv(OUT + 'F2_prevent_categories.csv', index=False)

json.dump(summ, open(OUT + 'F2_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
print(on.round(3).to_string())
print(WH.round(2).to_string())
print(KR[KR.stage == 3].round(3).to_string())
print(PC)
