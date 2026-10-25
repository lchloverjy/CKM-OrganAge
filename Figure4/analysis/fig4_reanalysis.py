"""Figure 4 extended analyses: CKM stage x multi-organ ageing burden -> incident disease.

Prospective sub-cohort, baseline-verified CKM stage, model a covariates; burden as in Fig. 2 (extreme ageing =
age gap above the 90th percentile of stage 0 participants of the same sex and age band; six target organs).

(1) standardised cumulative incidence of the CKM composite for 9 stage x burden groups (0-13 years, 200 bootstraps)
(2) dose-response: 10-year standardised risk for 0, 1, 2 and >=3 aged organs within stage; HR per aged organ
(3) outcome-specific HRs (1 and >=2 vs 0 aged organs, stage-adjusted), 10-year risk differences by stage,
    multiplicative interaction with stage and age-equivalent years
(4) specification analysis of the HR for >=2 vs 0 aged organs (CKM composite)
(5) 12-year state occupation (free / one system / >=2 systems) by stage x burden (Kaplan-Meier)
"""
import sys, time, json, warnings
sys.path.insert(0, './analysis'); sys.path.insert(0, './code')
from v3_common import *
from importlib import import_module
from patsy import dmatrix
import statsmodels.api as sm
from scipy.stats import norm, chi2
from lifelines import KaplanMeierFitter
warnings.filterwarnings('ignore')
R4 = './results/v4/'; os.makedirs(R4, exist_ok=True)
t0 = time.time()
NB = 200
OUTS = ['CKMC', 'HF', 'ASCVD', 'CKD', 'T2D', 'CLD', 'DEM']
SGL = ['0', '1', '2-3']

# ---------------------------------------------------------------- alternative burden definitions (cross-sectional cohort)
L1 = import_module('02_layer1'); L2 = import_module('03_layer2')
a = L1.load().reset_index(drop=True)
bu = pd.read_parquet(D + 'burden.parquet'); a = a.merge(bu, on='eid', how='left')
F = L2.extreme_flags(a)
assert (F[L2.TARGET].sum(1).values == a.burden_t.values).all()
a['cysc_f'] = a.cysc.fillna(a.cysc.median())
Z = dmatrix('cr(egfr, df=4) + cr(cysc_f, df=4)', a, return_type='dataframe')
ar = a.copy()
for o in L2.ORG10:
    g = a['gap_' + o]; ar['gap_' + o] = g - sm.OLS(g, Z).fit().fittedvalues
Fr = L2.extreme_flags(ar)
Fsd = L2.extreme_flags(a, sd_rule=True)
Fpo = L2.extreme_flags(a, stratified=False)
NOK = [o for o in L2.TARGET if o != 'Kidney']
ALT = pd.DataFrame({'eid': a.eid,
                    'cnt_prim': F[L2.TARGET].sum(1), 'cnt_nokid': F[NOK].sum(1),
                    'cnt_hkb': F[['Heart', 'Kidney', 'Brain']].sum(1), 'cnt_all10': F[L2.ORG10].sum(1),
                    'cnt_resid': Fr[L2.TARGET].sum(1), 'cnt_sd15': Fsd[L2.TARGET].sum(1), 'cnt_pooled': Fpo[L2.TARGET].sum(1)})
print('alt burden built', round(time.time() - t0))

# ---------------------------------------------------------------- prospective sub-cohort
def prep(stage_col='stage_bv'):
    s = subcohort(stage_col).merge(ALT, on='eid', how='left')
    assert (s.cnt_prim.values == s.burden_t.values).all()
    s['sg'] = np.where(s.stage_use >= 2, 2, s.stage_use).astype(int)
    s['b1'] = (s.bur_c == 1).astype(float); s['b2'] = (s.bur_c == 2).astype(float)
    s['b4'] = np.minimum(s.burden_t, 3).astype(int)
    s['grp'] = s.sg * 3 + s.bur_c.astype(int)
    for g in range(1, 9):
        s[f'g{g}'] = (s.grp == g).astype(float)
    s['grp4'] = s.sg * 4 + s.b4
    for g in range(1, 12):
        s[f'h{g}'] = (s.grp4 == g).astype(float)
    return s
s = prep(); N = len(s)
GD = [f'g{g}' for g in range(1, 9)]; HD = [f'h{g}' for g in range(1, 12)]
GL = {g: f'S{SGL[g // 3]}/B{["0", "1", ">=2"][g % 3]}' for g in range(9)}
HL = {g: f'S{SGL[g // 4]}/N{["0", "1", "2", ">=3"][g % 4]}' for g in range(12)}

def fit_std(d, tcol, ecol, dummies, covs, times):
    """Cox g-computation: standardised cumulative incidence at `times` for each dummy setting (row 0 = reference)."""
    X = d[dummies + covs].values.astype(float); T = d[tcol].values; E = d[ecol].values
    f = cox_fit(T, E, X); b = f['beta']; mu = X.mean(0)
    H0 = breslow_H0(T, E, (X - mu) @ b, times)
    lp_cov = (X[:, len(dummies):] - mu[len(dummies):]) @ b[len(dummies):]
    out = np.empty((len(dummies) + 1, len(times)))
    for g in range(len(dummies) + 1):
        shift = -mu[:len(dummies)] @ b[:len(dummies)] + (b[g - 1] if g > 0 else 0.0)
        r = np.exp(lp_cov + shift)
        out[g] = 1 - np.exp(-np.outer(H0, r)).mean(1)
    return out

def boot_idx(n, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(NB)]

# ---------------------------------------------------------------- (1) standardised cumulative incidence, composite
k = 'CKMC'
d = s[['t_' + k, 'ev_' + k, 'grp'] + GD + COV_A].dropna().reset_index(drop=True)
grid = np.round(np.arange(0, 13.01, 0.25), 2)
est = fit_std(d, 't_' + k, 'ev_' + k, GD, COV_A, grid)
bs = np.array([fit_std(d.iloc[i], 't_' + k, 'ev_' + k, GD, COV_A, grid) for i in boot_idx(len(d), 41)])
cur = []
for g in range(9):
    for j, t in enumerate(grid):
        cur.append({'group': GL[g], 'stage': SGL[g // 3], 'burden': ['0', '1', '>=2'][g % 3], 't': t, 'cuminc': est[g, j],
                    'lo': np.quantile(bs[:, g, j], 0.025), 'hi': np.quantile(bs[:, g, j], 0.975)})
CUR = pd.DataFrame(cur); CUR.to_csv(R4 + 'F4_std_cuminc_composite.csv', index=False)
rk = []
for g in range(9):
    dg = d[d.grp == g]
    for t in [0, 4, 8, 12]:
        rk.append({'group': GL[g], 't': t, 'n_risk': int((dg['t_' + k] >= t).sum()), 'n': len(dg), 'events': int(dg['ev_' + k].sum())})
pd.DataFrame(rk).to_csv(R4 + 'F4_at_risk_composite.csv', index=False)
# HR >=2 vs 0 and 1 vs 0 within each stage (age/sex/... adjusted, stratum-specific models)
ws = []
for sg in range(3):
    dd = s[s.sg == sg]; cov = COV_A + (['st3'] if sg == 2 else [])
    r, _ = cox_hr(dd, 't_' + k, 'ev_' + k, ['b1', 'b2'], cov); r['stage'] = SGL[sg]; ws.append(r)
WS = pd.concat(ws, ignore_index=True); WS.to_csv(R4 + 'F4_within_stage_hr_composite.csv', index=False)
print(WS[['stage', 'term', 'hr', 'lo', 'hi', 'p', 'n', 'events']].round(4).to_string()); print('(1) done', round(time.time() - t0))

# ---------------------------------------------------------------- (2) dose-response, composite
d = s[['t_' + k, 'ev_' + k, 'grp4'] + HD + COV_A].dropna().reset_index(drop=True)
est = fit_std(d, 't_' + k, 'ev_' + k, HD, COV_A, [10.0])[:, 0]
bs = np.array([fit_std(d.iloc[i], 't_' + k, 'ev_' + k, HD, COV_A, [10.0])[:, 0] for i in boot_idx(len(d), 43)])
dr = []
for g in range(12):
    cell = d.grp4 == g
    dr.append({'group': HL[g], 'stage': SGL[g // 4], 'n_aged': ['0', '1', '2', '>=3'][g % 4], 'risk10': est[g],
               'lo': np.quantile(bs[:, g], 0.025), 'hi': np.quantile(bs[:, g], 0.975), 'n': int(cell.sum()),
               'events': int(d.loc[cell, 'ev_' + k].sum())})
DR = pd.DataFrame(dr); DR.to_csv(R4 + 'F4_dose_response_risk10.csv', index=False); print(DR.round(4).to_string())
s['n2'] = (s.b4 == 2).astype(float); s['n3'] = (s.b4 == 3).astype(float); s['n1'] = (s.b4 == 1).astype(float)
trend = []
for kk in OUTS:
    r, _ = cox_hr(s, 't_' + kk, 'ev_' + kk, ['n1', 'n2', 'n3'], COV_A + STG); r['outcome'] = kk; r['model'] = 'categorical'; trend.append(r)
    r, _ = cox_hr(s, 't_' + kk, 'ev_' + kk, ['b4'], COV_A + STG); r['outcome'] = kk; r['model'] = 'per aged organ (0-3+)'; trend.append(r)
TR = pd.concat(trend, ignore_index=True); TR.to_csv(R4 + 'F4_dose_response_hr.csv', index=False)
print(TR[TR.outcome == 'CKMC'][['term', 'hr', 'lo', 'hi', 'p']].round(4).to_string()); print('(2) done', round(time.time() - t0))

# ---------------------------------------------------------------- (3) outcome-specific HR, RD by stage, interaction, age equivalence
COV_AL = [c for c in COV_A if c != 'age2']
s['sg1'] = (s.sg == 1).astype(float); s['sg2'] = (s.sg == 2).astype(float)
s['b2_sg1'] = s.b2 * s.sg1; s['b2_sg2'] = s.b2 * s.sg2; s['b1_sg1'] = s.b1 * s.sg1; s['b1_sg2'] = s.b1 * s.sg2
oc, rd = [], []
for kk in OUTS:
    tc, ec = 't_' + kk, 'ev_' + kk
    r, _ = cox_hr(s, tc, ec, ['b1', 'b2'], COV_A + STG)
    # multiplicative interaction with stage group (Wald, 2 df for >=2 aged organs)
    dd = s[[tc, ec, 'b1', 'b2', 'b2_sg1', 'b2_sg2', 'b1_sg1', 'b1_sg2'] + STG + COV_A].dropna()
    X = dd[['b2_sg1', 'b2_sg2', 'b1_sg1', 'b1_sg2', 'b1', 'b2'] + STG + COV_A].values
    f = cox_fit(dd[tc].values, dd[ec].values, X)
    bi = f['beta'][:2]; Vi = f['V'][:2, :2]; w = float(bi @ np.linalg.solve(Vi, bi)); p_int = float(chi2.sf(w, 2))
    # age-equivalent years for >=2 aged organs
    def age_eq(dd_):
        ff = cox_fit(dd_[tc].values, dd_[ec].values, dd_[['b2', 'b1', 'age'] + STG + [c for c in COV_AL if c != 'age']].values)
        return ff['beta'][0] / ff['beta'][2], ff['beta'][2]
    da = s[[tc, ec, 'b1', 'b2', 'age'] + STG + COV_AL].dropna().loc[:, lambda x: ~x.columns.duplicated()].reset_index(drop=True)
    ae, bage = age_eq(da)
    # 9-group standardised 10-year risk -> risk differences by stage
    d9 = s[[tc, ec, 'grp'] + GD + COV_A].dropna().reset_index(drop=True)
    est = fit_std(d9, tc, ec, GD, COV_A, [10.0])[:, 0]
    idx = boot_idx(len(d9), 47)
    bsr = np.array([fit_std(d9.iloc[i], tc, ec, GD, COV_A, [10.0])[:, 0] for i in idx])
    bae = np.array([age_eq(da.iloc[i])[0] for i in boot_idx(len(da), 53)])
    oc.append({'outcome': kk, 'hr_b1': r.hr[0], 'lo_b1': r.lo[0], 'hi_b1': r.hi[0], 'p_b1': r.p[0],
               'hr_b2': r.hr[1], 'lo_b2': r.lo[1], 'hi_b2': r.hi[1], 'p_b2': r.p[1], 'n': int(r.n[0]), 'events': int(r.events[0]),
               'p_int_stage_mult': p_int, 'age_equiv_years': ae, 'ae_lo': np.quantile(bae, 0.025), 'ae_hi': np.quantile(bae, 0.975),
               'hr_per_year_age': float(np.exp(bage))})
    for sg in range(3):
        g0, g1, g2 = sg * 3, sg * 3 + 1, sg * 3 + 2
        for lab, gg in [('>=2', g2), ('1', g1)]:
            v = est[gg] - est[g0]; b_ = bsr[:, gg] - bsr[:, g0]
            cell2 = d9.grp == gg
            rd.append({'outcome': kk, 'stage': SGL[sg], 'contrast': lab + ' vs 0', 'risk_ref': est[g0], 'risk_exp': est[gg],
                       'rd': v, 'lo': np.quantile(b_, 0.025), 'hi': np.quantile(b_, 0.975),
                       'n_exp': int(cell2.sum()), 'ev_exp': int(d9.loc[cell2, ec].sum()),
                       'n_ref': int((d9.grp == g0).sum()), 'ev_ref': int(d9.loc[d9.grp == g0, ec].sum())})
    # additive interaction: RD(stage 2-3) - RD(stage 0), >=2 vs 0
    dd_ = (est[8] - est[6]) - (est[2] - est[0]); bb = (bsr[:, 8] - bsr[:, 6]) - (bsr[:, 2] - bsr[:, 0])
    oc[-1].update({'rd_diff_23_vs_0': dd_, 'rd_diff_lo': np.quantile(bb, 0.025), 'rd_diff_hi': np.quantile(bb, 0.975),
                   'p_rd_diff_boot': float(2 * min((bb <= 0).mean(), (bb >= 0).mean()))})
    print(kk, 'done', round(time.time() - t0))
OC = pd.DataFrame(oc); OC.to_csv(R4 + 'F4_outcome_specific.csv', index=False)
RD = pd.DataFrame(rd); RD.to_csv(R4 + 'F4_risk_difference_by_stage.csv', index=False)
print(OC.round(3).to_string()); print(RD[RD.contrast == '>=2 vs 0'].round(4).to_string()); print('(3) done', round(time.time() - t0))

# ---------------------------------------------------------------- (4) specification analysis, composite, >=2 vs 0 aged organs
k = 'CKMC'; tc, ec = 't_' + k, 'ev_' + k
KID = [f'egfrcc_s{j}' for j in range(3)] + ['uacr_a2', 'uacr_a3', 'uacr_obs']
MODD = DEF + [f'egfrcc_s{j}' for j in range(3)] + ['uacr_a2', 'uacr_a3', 'nppb', 'tnni3_p']
def spec(d, cnt_col, covs, group, label, kmin=2):
    d = d.copy(); c = d[cnt_col]
    d['x1'] = ((c >= 1) & (c < kmin)).astype(float); d['x2'] = (c >= kmin).astype(float)
    terms = ['x1', 'x2'] if kmin > 1 else ['x2']
    r, _ = cox_hr(d, tc, ec, terms, covs)
    rr = r[r.term == 'x2'].iloc[0]
    return {'group': group, 'specification': label, 'hr': rr.hr, 'lo': rr.lo, 'hi': rr.hi, 'p': rr.p,
            'n_exposed': int(((d.x2 == 1) & d[ec].notna()).sum()), 'ev_exposed': int(d.loc[d.x2 == 1, ec].sum()),
            'n_ref': int(((c == 0) & d[ec].notna()).sum()), 'ev_ref': int(d.loc[c == 0, ec].sum()),
            'pct_exposed': float((d.loc[d[ec].notna(), 'x2'] == 1).mean() * 100)}
SP = []
B = COV_A + STG
SP.append(spec(s, 'cnt_prim', B, 'Primary', 'Primary analysis'))
SP.append(spec(s, 'cnt_prim', B + DEF, 'Adjustment', '+ Stage-defining traits'))
SP.append(spec(s, 'cnt_prim', B + KID, 'Adjustment', '+ eGFRcr-cys splines and UACR'))
SP.append(spec(s, 'cnt_prim', B + ['gap_Kidney'], 'Adjustment', '+ Kidney age gap (continuous)'))
SP.append(spec(s, 'cnt_prim', B + ['log_prevent'], 'Adjustment', '+ PREVENT risk'))
SP.append(spec(s, 'cnt_prim', B + MODD, 'Adjustment', '+ Model d covariates'))
SP.append(spec(s, 'cnt_nokid', B, 'Definition', 'Excluding the kidney clock (>=2 of 5)'))
SP.append(spec(s, 'cnt_hkb', B, 'Definition', 'Heart, kidney and brain only (>=2 of 3)'))
SP.append(spec(s, 'cnt_all10', B, 'Definition', 'All ten organ clocks (>=2 of 10)'))
SP.append(spec(s, 'cnt_resid', B, 'Definition', 'Kidney-function-residualised age gaps'))
SP.append(spec(s, 'cnt_sd15', B, 'Definition', 'Threshold: stage 0 mean + 1.5 s.d.'))
SP.append(spec(s, 'cnt_pooled', B, 'Definition', 'Threshold: pooled 90th percentile'))
so = prep('stage_orig')
SP.append(spec(so, 'cnt_prim', COV_A + STG, 'Cohort', 'Original stage assignment'))
s2 = s[(s[tc] > 2) | (s[ec] == 0)]
SP.append(spec(s2, 'cnt_prim', B, 'Cohort', 'Excluding events in first 2 years'))
for sx, lab in [(1, 'Men'), (0, 'Women')]:
    SP.append(spec(s[s.male == sx], 'cnt_prim', [c for c in B if c != 'male'], 'Cohort', lab))
SPD = pd.DataFrame(SP); SPD.to_csv(R4 + 'F4_specification_composite.csv', index=False)
print(SPD.round(3).to_string()); print('(4) done', round(time.time() - t0))

# ---------------------------------------------------------------- (5) 12-year state occupation by stage x burden
TS = np.column_stack([np.where(s['ev_SYS_' + x] == 1, s['t_SYS_' + x], np.inf) for x in SYSTEMS])
srt = np.sort(TS, axis=1)
s['t1'] = np.where(np.isfinite(srt[:, 0]), srt[:, 0], s.fu_years); s['e1'] = np.isfinite(srt[:, 0]).astype(float)
s['t2'] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], s.fu_years); s['e2'] = np.isfinite(srt[:, 1]).astype(float)
occ = []
for g in range(9):
    dg = s[s.grp == g]
    S1 = float(KaplanMeierFitter().fit(dg.t1, dg.e1).survival_function_at_times(12.0).values[0])
    S2 = float(KaplanMeierFitter().fit(dg.t2, dg.e2).survival_function_at_times(12.0).values[0])
    occ.append({'group': GL[g], 'stage': SGL[g // 3], 'burden': ['0', '1', '>=2'][g % 3], 'n': len(dg),
                'p_free': S1, 'p_one': S2 - S1, 'p_multi': 1 - S2, 'n_first': int(dg.e1.sum()), 'n_second': int(dg.e2.sum())})
OCC = pd.DataFrame(occ); OCC.to_csv(R4 + 'F4_state_occupation_12y.csv', index=False); print(OCC.round(4).to_string())
print('A->B', int(s.e1.sum()), 'B->C', int(s.e2.sum()), 'N', N)
print('total', round(time.time() - t0))
