"""Head-to-head comparison of organ proteomic clocks with single plasma biomarkers (pre-specified plan:
/home/ubuntu/work/v5/h2h/00_正面比较_预设分析方案.md).

Cohort, outcomes, censoring and the base model are those of Figs 3-5 (n = 40,674; baseline-verified CKM stage;
hospital inpatient + death registry to 30 Sep 2021).  Base model = Fig. 3 model b (COV_A + CKM stage).

A1  HR per s.d. of each exposure (alone and with the paired clock, mutually adjusted).
A2  Five-fold cross-fitted Harrell's C (2 repeats); delta C with 500 paired bootstrap resamples of the
    out-of-fold linear predictors (identical procedure to Fig. 5).
A3  10-year Aalen-Johansen risk (death competing) by aged clock x elevated marker (same 90th-centile rule
    in stage 0 by sex and age band for both).
A4  Multi-organ: CKM composite and >=2 newly affected organ systems; burden vs GDF15 vs six-marker panel,
    over model b and over guideline model G.
A5  Follow-up split at 5 years (landmark; identical partial likelihood to episode splitting).
S1  Exact-zero protein values (median-imputed missing) set to missing; complete-case.
S2  Model c adjustment (adds stage-defining traits).
Run from /home/ubuntu/work/v5/run.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import sys, json, time
sys.path.insert(0, './analysis'); sys.path.insert(0, './code')
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from multiprocessing import Pool
from v3_common import *
from fastc import cindex as fc
import pyreadr

OUT = '/home/ubuntu/work/v5/h2h/results/'; os.makedirs(OUT, exist_ok=True)
RDS = '/home/ubuntu/upload/ukb_protein_wide_ins0_final.rds'
t0 = time.time()
H = 10.0
NBOOT = 500
def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

# ------------------------------------------------------------------ cohort
s = subcohort()
B = COV_A + STG
C_MODEL = B + DEF
G = COV_A + STG + ['log_prevent'] + [f'egfrcc_s{j}' for j in range(3)] + ['uacr_a2', 'uacr_a3', 'nppb', 'tnni3_p',
                                                                         'htn_med', 'statin']
if 'died' not in s.columns:
    s['died'] = s['ev_DEATH']
# >=2 newly affected organ systems (identical to Fig. 6)
TS = np.column_stack([np.where(s['ev_SYS_' + k] == 1, s['t_SYS_' + k], np.inf) for k in SYSTEMS])
second = np.sort(TS, 1)[:, 1]
s['ev_MSYS'] = np.isfinite(second).astype(float); s['t_MSYS'] = np.where(np.isfinite(second), second, s.fu_years)
s['bur1'] = (s.burden_t == 1).astype(float); s['bur2'] = (s.burden_t >= 2).astype(float)

# ------------------------------------------------------------------ proteins
PROT = ['NTproBNP', 'NPPB', 'TNNI3', 'GDF15', 'NEFL', 'GFAP', 'CST3']
P = list(pyreadr.read_r(RDS).values())[0][['eid'] + PROT]
P['eid'] = pd.to_numeric(P.eid).astype(int)
s = s.merge(P, on='eid', how='left', suffixes=('', '_rds'))
assert s[PROT].notna().all().all() and len(s) == 40674
zero_share = {p: float((s[p] == 0).mean()) for p in PROT}
log('n', len(s), 'exact-zero shares', {k: round(v, 4) for k, v in zero_share.items()})
def zs(x):
    x = pd.Series(x, index=s.index, dtype=float); return (x - x.mean()) / x.std()
MK = {}
for p in PROT:
    s['m_' + p] = zs(s[p]); MK['m_' + p] = p
    s['m0_' + p] = zs(s[p].where(s[p] != 0))          # S1: exact-zero -> missing
s['m_cysc'] = zs(np.log(s.cysc)); s['m0_cysc'] = s['m_cysc']
s['m_egfrcc'] = -zs(s.egfr_cc); s['m0_egfrcc'] = s['m_egfrcc']      # per s.d. LOWER eGFRcr-cys
LAB = {'gap_Heart': 'Heart age gap', 'gap_Kidney': 'Kidney age gap', 'gap_Brain': 'Brain age gap',
       'gap_Organismal': 'Organismal age gap', 'm_NTproBNP': 'NT-proBNP (Olink)', 'm_NPPB': 'NPPB (Olink)',
       'm_TNNI3': 'Troponin I (TNNI3)', 'm_GDF15': 'GDF15', 'm_NEFL': 'NfL (NEFL)', 'm_GFAP': 'GFAP',
       'm_CST3': 'Cystatin C (Olink CST3)', 'm_cysc': 'Cystatin C (clinical, log)',
       'm_egfrcc': 'eGFRcr-cys (per s.d. lower)', 'burden': 'Aged-organ burden (1, >=2)',
       'panel6': 'Six-marker panel'}
COMPONENT = {('HF', 'm_NPPB'): 'clock component', ('HF', 'm_NTproBNP'): 'same gene product as clock component NPPB',
             ('DEM', 'm_NEFL'): 'clock component', ('DEM', 'm_GFAP'): 'clock component'}
PAIRS = {'HF': ('gap_Heart', ['m_NTproBNP', 'm_NPPB', 'm_TNNI3', 'm_GDF15'], ['m_NTproBNP', 'm_TNNI3']),
         'CKD': ('gap_Kidney', ['m_cysc', 'm_egfrcc', 'm_CST3', 'm_GDF15'], ['m_egfrcc', 'm_CST3']),
         'DEM': ('gap_Brain', ['m_NEFL', 'm_GFAP', 'm_GDF15'], ['m_NEFL', 'm_GFAP'])}
PANEL6 = ['m_NTproBNP', 'm_TNNI3', 'm_cysc', 'm_NEFL', 'm_GFAP', 'm_GDF15']

# ------------------------------------------------------------------ helpers
def frame(k, cols):
    tc, ec = 't_' + k, 'ev_' + k
    return s[['eid', tc, ec, 'died', 'age', 'male', 'stage_use'] +
             [c for c in dict.fromkeys(cols) if c not in ('age', 'male')]].dropna().reset_index(drop=True)
def hr_rows(d, k, expo, covs, tag):
    r, _ = cox_hr(d, 't_' + k, 'ev_' + k, expo, covs)
    r['outcome'] = k; r['analysis'] = tag; r['label'] = r.term.map(lambda x: LAB.get(x, x))
    return r
def make_folds(E, seed):
    rng = np.random.default_rng(seed); f = np.zeros(len(E), int)
    for v in [0, 1]:
        ix = np.where(E == v)[0]; rng.shuffle(ix); f[ix] = np.arange(len(ix)) % 5
    return f
def oof(X, T, E, folds):
    lp = np.zeros(len(T))
    for k in range(5):
        tr = folds != k; te = folds == k
        f = cox_fit(T[tr], E[tr], X[tr]); mu = X[tr].mean(0)
        lp[te] = (X[te] - mu) @ f['beta']
    return lp
def cross_c(d, k, models, comps, seed_boot):
    T = d['t_' + k].values; E = d['ev_' + k].values.astype(int); n = len(d)
    cs = {m: [] for m in models}; LP = {}
    for rep in range(2):
        folds = make_folds(E, 100 + rep)
        for m, cols in models.items():
            lp = oof(d[cols].values, T, E, folds); cs[m].append(fc(T, lp, E))
            if rep == 0:
                LP[m] = lp
    rng = np.random.default_rng(seed_boot); boot = {c: [] for c in comps}; bC = {m: [] for m in models}
    for b in range(NBOOT):
        ix = rng.integers(0, n, n)
        cb = {m: fc(T[ix], LP[m][ix], E[ix]) for m in models}
        for m in models:
            bC[m].append(cb[m])
        for new, base in comps:
            boot[(new, base)].append(cb[new] - cb[base])
    Cm = pd.DataFrame([{'outcome': k, 'model': m, 'n': n, 'events': int(E.sum()), 'C': np.mean(cs[m]),
                        'lo': np.quantile(bC[m], 0.025), 'hi': np.quantile(bC[m], 0.975)} for m in models])
    rows = []
    for new, base in comps:
        bb = np.array(boot[(new, base)]); dC = np.mean(cs[new]) - np.mean(cs[base]); se = bb.std(ddof=1)
        rows.append({'outcome': k, 'model': new, 'baseline': base, 'n': n, 'events': int(E.sum()),
                     'C_base': np.mean(cs[base]), 'C_new': np.mean(cs[new]), 'dC': dC,
                     'lo': np.quantile(bb, 0.025), 'hi': np.quantile(bb, 0.975), 'se_boot': se,
                     'p': 2 * norm.sf(abs(dC / se)) if se > 0 else np.nan})
    return Cm, pd.DataFrame(rows)
def aj_cif(t, e, dd, horizon=H):
    ut, inv = np.unique(t, return_inverse=True)
    ne = np.bincount(inv, weights=e); nd = np.bincount(inv, weights=dd); nn = np.bincount(inv)
    r = len(t) - np.r_[0, np.cumsum(nn)[:-1]]
    S = np.cumprod(1 - (ne + nd) / r); Sm = np.r_[1.0, S[:-1]]
    F = np.cumsum(Sm * ne / r)
    j = np.searchsorted(ut, horizon, side='right') - 1
    return float(F[j]) if j >= 0 else 0.0
def cif_ci(d, k, nb=200, seed=0):
    t = d['t_' + k].values.astype(float); e = d['ev_' + k].values.astype(float)
    dd = ((e == 0) & (d.died.values == 1)).astype(float)
    est = 100 * aj_cif(t, e, dd)
    r = np.random.default_rng(seed); v = []
    for _ in range(nb):
        ix = r.integers(0, len(t), len(t)); v.append(100 * aj_cif(t[ix], e[ix], dd[ix]))
    return est, np.percentile(v, 2.5), np.percentile(v, 97.5)
AGE_BANDS = [0, 50, 55, 60, 65, 100]
band_key = pd.cut(s.age, AGE_BANDS, right=False, labels=False).astype(int).astype(str) + '_' + s.male.astype(int).astype(str)
def high_flag(col, ref_stage):
    g = s[col]; ref = s[ref_stage] == 0
    thr = band_key.map(g[ref].groupby(band_key[ref]).quantile(0.90))
    return (g > thr).astype(int)

# which stage column reproduces the published burden?
for sc in ['stage', 'stage_bv', 'stage_use']:
    if sc in s.columns:
        bt = sum(high_flag('gap_' + o, sc) for o in TARGET)
        agree = float((bt == s.burden_t).mean())
        log('burden reproduction with', sc, round(agree, 4))
        if agree > 0.999:
            REFSTAGE = sc; break
else:
    raise SystemExit('cannot reproduce burden_t')
for c in ['gap_Heart', 'gap_Kidney', 'gap_Brain', 'gap_Organismal'] + list(LAB.keys()):
    if c in s.columns and (c.startswith('gap_') or c.startswith('m_')):
        s['hi_' + c] = high_flag(c, REFSTAGE)

# ------------------------------------------------------------------ descriptive correlations
CORR_ROWS = []
for a_ in ['gap_Heart', 'gap_Kidney', 'gap_Brain', 'gap_Organismal']:
    for b_ in ['m_NTproBNP', 'm_NPPB', 'm_TNNI3', 'm_cysc', 'm_egfrcc', 'm_CST3', 'm_NEFL', 'm_GFAP', 'm_GDF15']:
        rho = spearmanr(s[a_], s[b_]).correlation
        CORR_ROWS.append({'clock': a_, 'marker': b_, 'rho': rho, 'clock_label': LAB[a_], 'marker_label': LAB[b_]})
pd.DataFrame(CORR_ROWS).to_csv(OUT + 'H0_correlations.csv', index=False)
log('correlations done')

# ------------------------------------------------------------------ per-organ analyses (A1, A2, A3, A5, S1, S2)
def run_pair(k):
    clock, markers, panel = PAIRS[k]
    out = {}
    d = frame(k, B + DEF + [clock] + markers + ['hi_' + clock] + ['hi_' + m for m in markers])
    # A1: HR single and mutually adjusted (model b); S2 model c
    R = []
    for x in [clock] + markers:
        R.append(hr_rows(d, k, [x], B, 'single, model b'))
        R.append(hr_rows(d, k, [x], C_MODEL, 'single, model c'))
    for m in markers:
        r = hr_rows(d, k, [clock, m], B, 'mutual, model b'); r['pair_with'] = m; R.append(r)
        r = hr_rows(d, k, [clock, m], C_MODEL, 'mutual, model c'); r['pair_with'] = m; R.append(r)
    r = hr_rows(d, k, [clock] + panel, B, 'clock + organ panel, model b'); r['pair_with'] = '+'.join(panel); R.append(r)
    out['hr'] = pd.concat(R, ignore_index=True)
    # A2: cross-fitted C
    models = {'b': B, 'b+clock': B + [clock], 'b+panel': B + panel, 'b+panel+clock': B + panel + [clock]}
    for m in markers:
        models['b+' + m] = B + [m]; models['b+clock+' + m] = B + [clock, m]
    comps = [('b+clock', 'b'), ('b+panel', 'b'), ('b+panel+clock', 'b+panel'), ('b+panel+clock', 'b+clock'),
             ('b+clock', 'b+panel')]
    for m in markers:
        comps += [('b+' + m, 'b'), ('b+clock', 'b+' + m), ('b+clock+' + m, 'b+' + m), ('b+clock+' + m, 'b+clock')]
    out['C'], out['dC'] = cross_c(d, k, models, comps, seed_boot=7)
    # A3: aged clock x elevated marker, 10-y AJ risk
    A = []
    for m in markers:
        for hc in [0, 1]:
            for hm in [0, 1]:
                dd = d[(d['hi_' + clock] == hc) & (d['hi_' + m] == hm)]
                est, lo, hi = cif_ci(dd, k, seed=hc * 2 + hm)
                A.append({'outcome': k, 'clock': clock, 'marker': m, 'clock_aged': hc, 'marker_high': hm,
                          'n': len(dd), 'events': int(dd['ev_' + k].sum()), 'risk10': est, 'lo': lo, 'hi': hi})
        # HR of aged clock among marker-normal, and of high marker among clock-normal (model b)
        for sub_col, expo, tag in [('hi_' + m, 'hi_' + clock, 'aged clock | marker normal'),
                                   ('hi_' + clock, 'hi_' + m, 'high marker | clock normal')]:
            dd = d[d[sub_col] == 0]
            r = hr_rows(dd, k, [expo], B, tag); r['marker'] = m; A.append(r.iloc[0].to_dict())
    out['aj'] = pd.DataFrame(A)
    # A5: follow-up split at 5 years
    TSR = []
    for x in [clock] + markers:
        d1 = d.copy(); d1['ev_' + k] = ((d1['ev_' + k] == 1) & (d1['t_' + k] <= 5)).astype(float)
        d1['t_' + k] = d1['t_' + k].clip(upper=5)
        r = hr_rows(d1, k, [x], B, '0-5 y'); TSR.append(r)
        d2 = d[d['t_' + k] > 5].copy(); d2['t_' + k] = d2['t_' + k] - 5
        r = hr_rows(d2, k, [x], B, '>5 y'); TSR.append(r)
    out['tsplit'] = pd.concat(TSR, ignore_index=True)
    # S1: exact zero -> missing (complete case on the marker), HR and head-to-head dC
    S1 = []; S1c = []
    for m in markers:
        m0 = m.replace('m_', 'm0_')
        dz = frame(k, B + [clock, m0])
        for x, tag in [([m0], 'single'), ([clock], 'single (same sample)'), ([clock, m0], 'mutual')]:
            r = hr_rows(dz, k, x, B, 'S1 ' + tag); r['marker'] = m; S1.append(r)
        mods = {'b+clock': B + [clock], 'b+m': B + [m0], 'b+clock+m': B + [clock, m0]}
        _, dc = cross_c(dz, k, mods, [('b+clock', 'b+m'), ('b+clock+m', 'b+m'), ('b+clock+m', 'b+clock')], 11)
        dc['marker'] = m; S1c.append(dc)
    out['s1_hr'] = pd.concat(S1, ignore_index=True); out['s1_dC'] = pd.concat(S1c, ignore_index=True)
    log(k, 'done; n', len(d), 'events', int(d['ev_' + k].sum()))
    return k, out

# ------------------------------------------------------------------ multi-organ (A4)
def run_multi(k):
    cols = G + PANEL6 + GAPS + ['bur1', 'bur2', 'gap_Organismal', 'm_NTproBNP', 'm_GDF15']
    d = frame(k, cols)
    Gx = [c for c in G]
    models = {'b': B, 'b+burden': B + ['bur1', 'bur2'], 'b+organismal': B + ['gap_Organismal'],
              'b+GDF15': B + ['m_GDF15'], 'b+panel6': B + PANEL6, 'b+panel6+burden': B + PANEL6 + ['bur1', 'bur2'],
              'b+gaps12': B + GAPS, 'b+panel6+gaps12': B + PANEL6 + GAPS,
              'G': Gx, 'G+burden': Gx + ['bur1', 'bur2'], 'G+GDF15+NfL+GFAP': Gx + ['m_GDF15', 'm_NEFL', 'm_GFAP'],
              'G+GDF15+NfL+GFAP+burden': Gx + ['m_GDF15', 'm_NEFL', 'm_GFAP', 'bur1', 'bur2'],
              'G+gaps12': Gx + GAPS, 'G+GDF15+NfL+GFAP+gaps12': Gx + ['m_GDF15', 'm_NEFL', 'm_GFAP'] + GAPS}
    comps = [('b+burden', 'b'), ('b+organismal', 'b'), ('b+GDF15', 'b'), ('b+panel6', 'b'), ('b+burden', 'b+GDF15'),
             ('b+burden', 'b+panel6'), ('b+panel6+burden', 'b+panel6'), ('b+gaps12', 'b+panel6'),
             ('b+panel6+gaps12', 'b+panel6'), ('b+gaps12', 'b'),
             ('G+burden', 'G'), ('G+GDF15+NfL+GFAP', 'G'), ('G+GDF15+NfL+GFAP+burden', 'G+GDF15+NfL+GFAP'),
             ('G+gaps12', 'G'), ('G+GDF15+NfL+GFAP+gaps12', 'G+GDF15+NfL+GFAP')]
    Cm, dC = cross_c(d, k, models, comps, seed_boot=13)
    R = [hr_rows(d, k, ['bur1', 'bur2'], B, 'burden, model b'),
         hr_rows(d, k, ['bur1', 'bur2'], B + PANEL6, 'burden, model b + six-marker panel'),
         hr_rows(d, k, ['m_GDF15'], B, 'GDF15, model b'),
         hr_rows(d, k, ['gap_Organismal'], B, 'organismal, model b'),
         hr_rows(d, k, ['gap_Organismal', 'm_GDF15'], B, 'organismal + GDF15, model b')]
    # absolute risk: burden x number of elevated single markers (0 vs >=1 of six)
    nh = sum(s.loc[d.index if False else s.index, 'hi_' + m] for m in PANEL6)
    dd_all = d.merge(pd.DataFrame({'eid': s.eid, 'n_high': nh.values}), on='eid')
    A = []
    for bc in [0, 1, 2]:
        for hm in [0, 1]:
            dd = dd_all[(np.minimum(dd_all.bur1 + 2 * dd_all.bur2, 2) == bc) & ((dd_all.n_high >= 1).astype(int) == hm)]
            est, lo, hi = cif_ci(dd, k, seed=bc * 2 + hm)
            A.append({'outcome': k, 'burden': bc, 'any_marker_high': hm, 'n': len(dd),
                      'events': int(dd['ev_' + k].sum()), 'risk10': est, 'lo': lo, 'hi': hi})
    log(k, 'multi done; n', len(d), 'events', int(d['ev_' + k].sum()))
    return k, {'C': Cm, 'dC': dC, 'hr': pd.concat(R, ignore_index=True), 'aj': pd.DataFrame(A)}

if __name__ == '__main__':
    with Pool(5) as pool:
        res_p = pool.map_async(run_pair, list(PAIRS))
        res_m = pool.map_async(run_multi, ['CKMC', 'MSYS'])
        RP = dict(res_p.get()); RM = dict(res_m.get())
    for key in ['hr', 'C', 'dC', 'aj', 'tsplit', 's1_hr', 's1_dC']:
        pd.concat([RP[k][key] for k in PAIRS], ignore_index=True).to_csv(OUT + f'H1_{key}.csv', index=False)
    for key in ['C', 'dC', 'hr', 'aj']:
        pd.concat([RM[k][key] for k in RM], ignore_index=True).to_csv(OUT + f'H4_multi_{key}.csv', index=False)
    # Holm across the pre-specified head-to-head comparisons (clock vs each organ marker)
    dC = pd.read_csv(OUT + 'H1_dC.csv')
    hh = dC[(dC.model == 'b+clock') & dC.baseline.str.startswith('b+m_')].copy()
    hh['p_holm'] = holm(hh.p.values)
    hh.to_csv(OUT + 'H1_head_to_head_holm.csv', index=False)
    json.dump({'n': len(s), 'zero_share': zero_share, 'refstage': REFSTAGE, 'nboot': NBOOT,
               'runtime_s': time.time() - t0}, open(OUT + 'H_meta.json', 'w'), indent=1)
    log('ALL DONE')
