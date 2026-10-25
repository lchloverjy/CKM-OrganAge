"""Common utilities for the v3 analyses (CKM -> organ age -> disease)."""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
D = './data/'; R = './results/v3/'
import os; os.makedirs(R, exist_ok=True)
ORG = ['Kidney', 'Artery', 'Heart', 'Liver', 'Pancreas', 'Immune', 'Brain', 'Lung', 'Intestine', 'Muscle',
       'Organismal', 'Conventional']
ORG10 = ORG[:10]
TARGET = ['Kidney', 'Artery', 'Heart', 'Liver', 'Pancreas', 'Brain']
GAPS = ['gap_' + o for o in ORG]
TIER1 = [('HF', 'Heart'), ('ASCVD', 'Artery'), ('CKD', 'Kidney'), ('T2D', 'Pancreas'), ('CLD', 'Liver'), ('DEM', 'Brain')]
TIER1_LAB = {'HF': 'Heart failure', 'ASCVD': 'ASCVD', 'CKD': 'Chronic kidney disease', 'T2D': 'Type 2 diabetes',
             'CLD': 'Chronic liver disease', 'DEM': 'All-cause dementia', 'CKMC': 'CKM multi-organ composite'}
SYSTEMS = ['Cardiac', 'Vascular', 'Kidney', 'Metabolic', 'Liver', 'Brain', 'Lung', 'Immune', 'Musculoskeletal',
           'Gastrointestinal', 'Eye', 'Cancer']
SYS_PAIR = {'Cardiac': 'Heart', 'Vascular': 'Artery', 'Kidney': 'Kidney', 'Metabolic': 'Pancreas', 'Liver': 'Liver',
            'Brain': 'Brain', 'Lung': 'Lung', 'Immune': 'Immune', 'Musculoskeletal': 'Muscle',
            'Gastrointestinal': 'Intestine'}
COV_A = ['age', 'age2', 'male', 'white', 'townsend', 'degree', 'smoke1', 'smoke2', 'alc1', 'alc2']
STG = ['st1', 'st2', 'st3']
DEF = ['bmi', 'waist', 'hba1c', 'sbp', 'log_tg_hdl', 'egfr']
LIFE = ['smoking_healthy', 'alcohol_healthy', 'diet_healthy', 'sleep_healthy', 'activity_healthy',
        'sedentary_healthy', 'social_healthy']


def egfr_crcys(scr_umol, cys, age, male):
    scr = scr_umol / 88.4
    k = np.where(male == 1, 0.9, 0.7); a = np.where(male == 1, -0.144, -0.219)
    e = (135 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -0.544 *
         np.minimum(cys / 0.8, 1) ** -0.323 * np.maximum(cys / 0.8, 1) ** -0.778 * 0.9961 ** age)
    return e * np.where(male == 1, 1.0, 0.963)


def ns_basis(x, knots_q=(0.05, 0.35, 0.65, 0.95), knots=None):
    """Restricted cubic spline basis (Harrell), returns k-2 non-linear terms plus linear."""
    x = np.asarray(x, float)
    t = np.quantile(x[~np.isnan(x)], knots_q) if knots is None else np.asarray(knots)
    k = len(t); out = [x]
    d = lambda u: np.maximum(u, 0) ** 3
    for j in range(k - 2):
        out.append((d(x - t[j]) - d(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2]) +
                    d(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(out)


def base_frame():
    a = pd.read_parquet(D + 'analysis.parquet')
    a = a[(a.has_gap == 1) & a.stage.isin([0, 1, 2, 3])].copy()
    for c in ['townsend', 'bmi', 'waist', 'hba1c', 'glucose', 'tg_hdl', 'egfr', 'crp', 'ldl', 'hdl', 'tg', 'sbp',
              'dbp', 'alt', 'ggt', 'cysc', 'creat', 'prevent10']:
        a[c] = a[c].fillna(a[c].median())
    for c in ['white', 'degree', 'smoke', 'alcohol', 'apoe4', 'htn_med', 'statin']:
        a[c] = a[c].fillna(a[c].mode()[0])
    a['age2'] = (a.age - 57) ** 2 / 10
    a['log_tg_hdl'] = np.log(a.tg_hdl); a['log_alt'] = np.log(a.alt); a['log_ggt'] = np.log(a.ggt)
    a['smoke1'] = (a.smoke == 1).astype(float); a['smoke2'] = (a.smoke == 2).astype(float)
    a['alc1'] = (a.alcohol == 1).astype(float); a['alc2'] = (a.alcohol == 2).astype(float)
    a['egfr_cc'] = egfr_crcys(a.creat.values, a.cysc.values, a.age.values, a.male.values)
    u = a.uacr
    a['uacr_a2'] = ((u >= 3) & (u < 30)).astype(float); a['uacr_a3'] = (u >= 30).astype(float)
    a['uacr_obs'] = u.notna().astype(float)
    S = ns_basis(a.egfr_cc.values)
    for j in range(S.shape[1]):
        a[f'egfrcc_s{j}'] = S[:, j]
    a['nppb'] = a.ntprobnp.fillna(a.ntprobnp.median()); a['tnni3_p'] = a.tnni3.fillna(a.tnni3.median())
    sdy = {}
    for o in ORG:
        y = a['gap_' + o]; sdy[o] = float(y.std())
        a['yr_' + o] = y; a['gap_' + o] = (y - y.mean()) / y.std()
    json.dump(sdy, open(R + 'gap_sd_years.json', 'w'), indent=1)
    b = pd.read_parquet(D + 'burden.parquet')
    a = a.merge(b[['eid', 'burden_t']], on='eid', how='left')
    if os.path.exists('./raw/lifestyle_category.csv'):
        L = pd.read_csv('./raw/lifestyle_category.csv').rename(columns={'Eid': 'eid'})
        a = a.merge(L, on='eid', how='left')
    else:                                   # v5: lifestyle file pending; lifestyle analyses (appendix) are skipped
        for c in LIFE:
            a[c] = np.nan
    a['life_n_obs'] = a[LIFE].notna().sum(axis=1)
    a['life_score'] = a[LIFE].sum(axis=1, min_count=7)
    return a


def subcohort(stage_col='stage_bv'):
    a = base_frame()
    O = pd.read_parquet(D + 'dx/outcomes_v3.parquet')
    s = a.merge(O, on='eid', how='inner')
    s['stage_use'] = s[stage_col]
    s['prevent_use'] = s.prevent_bv if stage_col == 'stage_bv' else s.prevent10
    s['log_prevent'] = np.log(s.prevent_use.clip(0.1))
    for k in range(1, 4):
        s[f'st{k}'] = (s.stage_use == k).astype(float)
    s['st23'] = (s.stage_use >= 2).astype(float)
    s['bur_c'] = np.minimum(s.burden_t, 2)
    return s


# ---------------------------------------------------------------- fast Cox (Breslow ties)
def _prep(T, E):
    o = np.argsort(-T, kind='mergesort')
    Ts = T[o]
    # index of the last element of each tie group in descending order
    _, first_idx, counts = np.unique(-Ts, return_index=True, return_counts=True)
    last = np.repeat(first_idx + counts - 1, counts)
    return o, last


def cox_fit(T, E, X, max_iter=40, tol=1e-9, ridge=0.0):
    T = np.asarray(T, float); E = np.asarray(E, float); X = np.asarray(X, float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    o, last = _prep(T, E)
    Z = Z[o]; Es = E[o]; n, p = Z.shape
    b = np.zeros(p); ll_old = -np.inf
    ev = Es == 1
    for it in range(max_iter):
        eta = Z @ b; m = eta.max(); w = np.exp(eta - m)
        S0 = np.cumsum(w)[last]
        S1 = np.cumsum(w[:, None] * Z, axis=0)[last]
        ll = np.sum(eta[ev] - m - np.log(S0[ev])) - 0.5 * ridge * b @ b
        Zb = S1[ev] / S0[ev, None]
        g = np.sum(Z[ev] - Zb, axis=0) - ridge * b
        # Hessian: sum over events of (S2/S0 - Zb Zb'); S2 term rewritten as sum_j w_j c_j Z_j Z_j'
        acc = np.zeros(n); np.add.at(acc, last[ev], 1.0 / S0[ev])
        cj = np.cumsum(acc[::-1])[::-1]
        H = (Z * (w * cj)[:, None]).T @ Z - Zb.T @ Zb + ridge * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        if ll < ll_old - 1e-8:
            b = b - step_prev * 0.5; step_prev *= 0.5; continue
        b_new = b + step; step_prev = step
        if np.abs(ll - ll_old) < tol:
            b = b_new; break
        ll_old = ll; b = b_new
    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        V = np.linalg.pinv(H)
    beta = b / sd; se = np.sqrt(np.diag(V)) / sd
    return {'beta': beta, 'se': se, 'V': V / np.outer(sd, sd), 'mu': mu, 'll': ll, 'n': n, 'events': int(E.sum())}


def cox_hr(df, tcol, ecol, expo, covs, extra=None):
    """HR for each exposure column in `expo` (list) adjusted for covs."""
    d = df[[tcol, ecol] + expo + covs].dropna()
    X = d[expo + covs].values
    f = cox_fit(d[tcol].values, d[ecol].values, X)
    out = []
    from scipy.stats import norm
    for i, e in enumerate(expo):
        b, s = f['beta'][i], f['se'][i]
        out.append({'term': e, 'beta': b, 'se': s, 'hr': np.exp(b), 'lo': np.exp(b - 1.96 * s),
                    'hi': np.exp(b + 1.96 * s), 'p': 2 * norm.sf(abs(b / s)), 'n': f['n'], 'events': f['events']})
    return pd.DataFrame(out), f


def breslow_H0(T, E, lp, times):
    """Baseline cumulative hazard at `times` given linear predictor lp (centred as supplied)."""
    o = np.argsort(T); T = T[o]; E = E[o]; w = np.exp(lp[o])
    rs = np.cumsum(w[::-1])[::-1]
    ut = T[E == 1]; inc = 1 / rs[E == 1]
    H = np.cumsum(inc)
    return np.array([H[ut <= t][-1] if np.any(ut <= t) else 0.0 for t in np.atleast_1d(times)])


def cindex(T, E, risk):
    from fastc import cindex as _fc   # v5: same conventions as lifelines concordance_index(T, -risk, E)
    return _fc(np.asarray(T, float), np.asarray(risk, float), np.asarray(E))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1); return out


def gap_sd_years():
    """SD (years) of each organ age gap in the proteomic sample, used to convert per-SD effects to years."""
    f = './results/v3/gap_sd_years.json'
    if os.path.exists(f):
        return json.load(open(f))
    ag = pd.read_excel('./raw/age_gap.xlsx')
    out = {}
    for o in ORG:
        cand = [c for c in ag.columns if c.lower().startswith(o.lower())]
        if cand:
            out[o] = float(ag[cand[0]].std())
    json.dump(out, open(f, 'w'), indent=1)
    return out
