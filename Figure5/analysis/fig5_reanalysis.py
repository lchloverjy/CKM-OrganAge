"""Figure 5 extended analyses (clinical value of organ age gaps over guideline-based assessment).

Re-uses the module-4 cohort, models and cross-fitting folds (m4.py) and adds
  A. for the CKM composite and six tier-1 outcomes:
     - delta C over model G for each single clock, for the aged-organ count and for all 12 clocks;
       delta C over model D for all 12 clocks (500 paired bootstrap resamples of out-of-fold predictions);
     - stage-specific delta C (stage 0, 1, 2-3) with bootstrap CIs;
     - case-capture curves in stages 1-2 (PREVENT, model G, model G + gaps);
     - observed 10-year risk among participants reclassified by the age gaps;
     - categorical and continuous NRI with bootstrap CIs;
     - difference in net benefit across thresholds with bootstrap CIs.
  B. for 46 tier-2 diseases, the negative-control outcome and 12 organ-system composites:
     delta C of the 12 gaps over model G and over the minimal model (model a covariates + stage), with bootstrap CIs.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, './analysis'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import *
from fastc import cindex as fc
from lifelines import KaplanMeierFitter
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool

RV4 = './results/v4/'; os.makedirs(RV4, exist_ok=True)
t0 = time.time()
s = subcohort()
G = COV_A + STG + ['log_prevent'] + [f'egfrcc_s{j}' for j in range(3)] + ['uacr_a2', 'uacr_a3', 'nppb', 'tnni3_p', 'htn_med', 'statin']
DX = {'CKMC': ['bmi', 'sbp', 'hba1c', 'log_tg_hdl'], 'HF': ['bmi', 'sbp', 'hba1c'], 'ASCVD': ['ldl', 'sbp', 'hdl'],
      'CKD': ['sbp', 'hba1c', 'bmi'], 'T2D': ['hba1c', 'bmi', 'log_tg_hdl', 'glucose'], 'CLD': ['log_alt', 'log_ggt', 'bmi'],
      'DEM': ['apoe4']}
PAIR = dict(TIER1); PAIR['CKMC'] = None
OUTS = ['CKMC'] + [k for k, _ in TIER1]
H = 10.0
s['sg'] = np.where(s.stage_use >= 2, 2, s.stage_use).astype(int)
s['bur1'] = (s.burden_t == 1).astype(float); s['bur2'] = (s.burden_t >= 2).astype(float)
THR = {k: ([0.05, 0.10] if k in ['CKMC', 'ASCVD'] else [0.025, 0.05]) for k in OUTS}
PTS = np.round(np.arange(0.005, 0.301, 0.005), 3)
FRAC = np.round(np.arange(0.01, 0.401, 0.01), 2)


def oof(d, tc, ec, cols, folds):
    lp = np.zeros(len(d)); risk = np.zeros(len(d))
    X = d[cols].values; T = d[tc].values; E = d[ec].values
    for k in range(5):
        tr = folds != k; te = folds == k
        f = cox_fit(T[tr], E[tr], X[tr]); b = f['beta']; mu = X[tr].mean(0)
        H0 = breslow_H0(T[tr], E[tr], (X[tr] - mu) @ b, [H])[0]
        lp[te] = (X[te] - mu) @ b; risk[te] = 1 - np.exp(-H0 * np.exp(lp[te]))
    return lp, risk


def make_folds(E, seed):
    rng = np.random.default_rng(seed); f = np.zeros(len(E), int)
    for v in [0, 1]:
        ix = np.where(E == v)[0]; rng.shuffle(ix); f[ix] = np.arange(len(ix)) % 5
    return f


def km_risk(T, E, h=H):
    if len(T) == 0:
        return np.nan, np.nan, np.nan
    k = KaplanMeierFitter().fit(T, E)
    sf = k.survival_function_.loc[:h].iloc[-1, 0]
    ci = k.confidence_interval_.loc[:h].iloc[-1]
    return 1 - sf, 1 - ci.iloc[1], 1 - ci.iloc[0]


def nb(risk, y, pt):
    n = len(y); sel = risk >= pt
    return (np.sum(sel & (y == 1)) - np.sum(sel & (y == 0)) * pt / (1 - pt)) / n


def capture(v, y, frac):
    o = np.argsort(-v); tot = max(1, y.sum())
    return np.array([y[o[:int(round(f * len(v)))]].sum() / tot for f in frac])


# ================================================================ part A: tier-1 outcomes and composite
def part_a(k):
    tc, ec = 't_' + k, 'ev_' + k
    cols_all = list(dict.fromkeys(G + DX[k] + GAPS))
    d = s[['eid', tc, ec, 'sg', 'male', 'age', 'stage_use', 'burden_t', 'log_prevent', 'bur1', 'bur2'] +
          [c for c in cols_all if c not in ['male', 'age', 'log_prevent']]].dropna().reset_index(drop=True)
    T, E = d[tc].values, d[ec].values.astype(int); n = len(d)
    y10 = ((E == 1) & (T <= H)).astype(int)
    models = {'G': G, 'G+gaps': G + GAPS, 'D': G + DX[k], 'D+gaps': G + DX[k] + GAPS, 'G+burden': G + ['bur1', 'bur2']}
    for o in ORG:
        models['G+' + o] = G + ['gap_' + o]
    cs = {m: [] for m in models}; P = {}
    for rep in range(2):
        folds = make_folds(E, 100 + rep)
        for m, cols in models.items():
            lp, rk = oof(d, tc, ec, cols, folds)
            cs[m].append(fc(T, lp, E))
            if rep == 0:
                P[m] = (lp, rk)
    comps = [(m, 'G') for m in models if m.startswith('G+')] + [('D+gaps', 'D')]
    rng = np.random.default_rng(9); boot = {c: [] for c in comps}
    for b in range(500):
        ix = rng.integers(0, n, n); cb = {}
        for m in models:
            cb[m] = fc(T[ix], P[m][0][ix], E[ix])
        for new, base in comps:
            boot[(new, base)].append(cb[new] - cb[base])
    disc = []
    for new, base in comps:
        bb = np.array(boot[(new, base)])
        disc.append({'outcome': k, 'n': n, 'events': int(E.sum()), 'events10': int(y10.sum()), 'model': new, 'baseline': base,
                     'added': 'All 12 clocks' if new in ['G+gaps', 'D+gaps'] else ('Aged-organ count' if new == 'G+burden' else new[2:]),
                     'paired': (new[2:] == PAIR[k]) if PAIR[k] else False,
                     'C_base': np.mean(cs[base]), 'C_new': np.mean(cs[new]), 'dC': np.mean(cs[new]) - np.mean(cs[base]),
                     'lo': np.quantile(bb, 0.025), 'hi': np.quantile(bb, 0.975), 'se_boot': bb.std(ddof=1)})
    # stage-specific delta C (G+gaps vs G), bootstrap within stratum
    rng = np.random.default_rng(21); stg = []
    for g in [0, 1, 2]:
        ii = np.where(d.sg.values == g)[0]
        if E[ii].sum() < 20:
            continue
        c0 = fc(T[ii], P['G'][0][ii], E[ii]); c1 = fc(T[ii], P['G+gaps'][0][ii], E[ii]); bb = []
        for b in range(200):
            jx = ii[rng.integers(0, len(ii), len(ii))]
            bb.append(fc(T[jx], P['G+gaps'][0][jx], E[jx]) - fc(T[jx], P['G'][0][jx], E[jx]))
        stg.append({'outcome': k, 'stage_group': ['Stage 0', 'Stage 1', 'Stage 2-3'][g], 'n': len(ii), 'events': int(E[ii].sum()),
                    'C_G': c0, 'C_Ggaps': c1, 'dC': c1 - c0, 'lo': np.quantile(bb, 0.025), 'hi': np.quantile(bb, 0.975)})
    # case capture in stages 1-2
    m12 = d.stage_use.isin([1, 2]).values; ii = np.where(m12)[0]; yy = y10[ii]
    sel = {'PREVENT': d.log_prevent.values[ii], 'Model G': P['G'][1][ii], 'Model G + age gaps': P['G+gaps'][1][ii]}
    if PAIR[k]:
        sel['Paired organ age gap'] = d['gap_' + PAIR[k]].values[ii]
    cap = []
    for lab, v in sel.items():
        cc = capture(v, yy, FRAC)
        for f, c in zip(FRAC, cc):
            cap.append({'outcome': k, 'selector': lab, 'fraction_selected': f, 'cases_captured': c, 'n_stage12': len(ii), 'cases10_stage12': int(yy.sum())})
    rng = np.random.default_rng(31); bd = {('Model G + age gaps', 'Model G'): [], ('Model G + age gaps', 'PREVENT'): []}
    n10 = int(round(0.10 * len(ii)))
    def top10(v, y):
        return y[np.argsort(-v)[:n10]].sum() / max(1, y.sum())
    point = {lab: top10(v, yy) for lab, v in sel.items()}
    for b in range(200):
        jx = rng.integers(0, len(ii), len(ii)); yb = yy[jx]
        cb = {lab: top10(v[jx], yb) for lab, v in sel.items()}
        for a_, b_ in bd:
            bd[(a_, b_)].append(cb[a_] - cb[b_])
    cap10 = []
    for (a_, b_), bb in bd.items():
        cap10.append({'outcome': k, 'comparison': f'{a_} minus {b_}', 'n_selected': n10, 'cases10_stage12': int(yy.sum()),
                      'capture_a': point[a_], 'capture_b': point[b_], 'diff': point[a_] - point[b_],
                      'lo': np.quantile(bb, 0.025), 'hi': np.quantile(bb, 0.975)})
    # reclassification: observed risk by movement within model G category
    thr = THR[k]; c0 = np.digitize(P['G'][1], thr); c1 = np.digitize(P['G+gaps'][1], thr); mv = np.sign(c1 - c0)
    cats = ['<' + f'{thr[0]*100:g}%', f'{thr[0]*100:g}–{thr[1]*100:g}%', '≥' + f'{thr[1]*100:g}%']
    rc = []
    for c in range(3):
        for mvv, lab in [(-1, 'Down'), (0, 'Unchanged'), (1, 'Up')]:
            ii2 = np.where((c0 == c) & (mv == mvv))[0]
            r, lo, hi = km_risk(T[ii2], E[ii2])
            rc.append({'outcome': k, 'G_category': cats[c], 'G_category_index': c, 'movement': lab, 'n': len(ii2),
                       'events10': int(y10[ii2].sum()) if len(ii2) else 0, 'obs10': r, 'obs10_lo': lo, 'obs10_hi': hi,
                       'pred_G': P['G'][1][ii2].mean() if len(ii2) else np.nan,
                       'pred_Ggaps': P['G+gaps'][1][ii2].mean() if len(ii2) else np.nan})
    # NRI with bootstrap CIs
    def nri(ix):
        y = y10[ix]; a0, a1 = c0[ix], c1[ix]; r0, r1 = P['G'][1][ix], P['G+gaps'][1][ix]
        up, dn = a1 > a0, a1 < a0; upc, dnc = r1 > r0, r1 < r0
        return np.array([up[y == 1].mean() - dn[y == 1].mean(), dn[y == 0].mean() - up[y == 0].mean(),
                         upc[y == 1].mean() - dnc[y == 1].mean(), dnc[y == 0].mean() - upc[y == 0].mean()])
    pt_ = nri(np.arange(n)); rng = np.random.default_rng(41)
    bb = np.array([nri(rng.integers(0, n, n)) for _ in range(200)])
    nr = []
    for j, lab in enumerate(['Categorical NRI, events', 'Categorical NRI, non-events', 'Continuous NRI, events', 'Continuous NRI, non-events']):
        nr.append({'outcome': k, 'component': lab, 'thresholds': str(thr), 'estimate': pt_[j],
                   'lo': np.quantile(bb[:, j], 0.025), 'hi': np.quantile(bb[:, j], 0.975)})
    for lab, idx in [('Categorical NRI, total', (0, 1)), ('Continuous NRI, total', (2, 3))]:
        tot = bb[:, idx[0]] + bb[:, idx[1]]
        nr.append({'outcome': k, 'component': lab, 'thresholds': str(thr), 'estimate': pt_[idx[0]] + pt_[idx[1]],
                   'lo': np.quantile(tot, 0.025), 'hi': np.quantile(tot, 0.975)})
    # net benefit difference with bootstrap CIs
    rng = np.random.default_rng(51); rG, rN = P['G'][1], P['G+gaps'][1]
    def dnb(ix):
        y = y10[ix]
        return np.array([nb(rN[ix], y, p) - nb(rG[ix], y, p) for p in PTS])
    base_ = dnb(np.arange(n)); bb = np.array([dnb(rng.integers(0, n, n)) for _ in range(200)])
    prev = y10.mean(); nbd = []
    for j, p in enumerate(PTS):
        nbd.append({'outcome': k, 'pt': p, 'nb_G': nb(rG, y10, p), 'nb_Ggaps': nb(rN, y10, p),
                    'nb_treat_all': prev - (1 - prev) * p / (1 - p),
                    'dnb_per1000': 1000 * base_[j], 'lo_per1000': 1000 * np.quantile(bb[:, j], 0.025),
                    'hi_per1000': 1000 * np.quantile(bb[:, j], 0.975)})
    return k, pd.DataFrame(disc), pd.DataFrame(stg), pd.DataFrame(cap), pd.DataFrame(cap10), pd.DataFrame(rc), pd.DataFrame(nr), pd.DataFrame(nbd)


# ================================================================ part B: tier-2 diseases, negative control, systems
OM = pd.read_csv('./results/V3_outcome_meta.csv')
KB = OM.key.tolist() + ['SYS_' + x for x in SYSTEMS]


def part_b(k):
    tc, ec = 't_' + k, 'ev_' + k
    out = {'outcome': k}
    # model G baseline (two cross-fitting repetitions, as for the tier-1 outcomes)
    d = s[[tc, ec] + G + GAPS].dropna().reset_index(drop=True); T, E = d[tc].values, d[ec].values.astype(int); n = len(d)
    c0s, c1s = [], []
    for rep in range(2):
        folds = make_folds(E, 100 + rep)
        lp0, _ = oof(d, tc, ec, G, folds); lp1, _ = oof(d, tc, ec, G + GAPS, folds)
        c0s.append(fc(T, lp0, E)); c1s.append(fc(T, lp1, E))
        if rep == 0:
            L0, L1 = lp0, lp1
    rng = np.random.default_rng(61); bb = []
    for b in range(200):
        ix = rng.integers(0, n, n); bb.append(fc(T[ix], L1[ix], E[ix]) - fc(T[ix], L0[ix], E[ix]))
    bb = np.array(bb)
    out.update({'n_G': n, 'events_G': int(E.sum()), 'C_G': np.mean(c0s), 'C_Ggaps': np.mean(c1s), 'dC_G': np.mean(c1s) - np.mean(c0s),
                'lo_G': np.quantile(bb, 0.025), 'hi_G': np.quantile(bb, 0.975), 'se_G': bb.std(ddof=1)})
    # minimal baseline (as in module 4: single split, fold seed 5)
    d = s[[tc, ec] + COV_A + STG + GAPS].dropna().reset_index(drop=True); T, E = d[tc].values, d[ec].values.astype(int); n = len(d)
    folds = make_folds(E, 5)
    lp0, _ = oof(d, tc, ec, COV_A + STG, folds); lp1, _ = oof(d, tc, ec, COV_A + STG + GAPS, folds)
    c0, c1 = fc(T, lp0, E), fc(T, lp1, E)
    rng = np.random.default_rng(62); bb = []
    for b in range(200):
        ix = rng.integers(0, n, n); bb.append(fc(T[ix], lp1[ix], E[ix]) - fc(T[ix], lp0[ix], E[ix]))
    bb = np.array(bb)
    out.update({'n_min': n, 'events_min': int(E.sum()), 'C_min': c0, 'C_mingaps': c1, 'dC_min': c1 - c0,
                'lo_min': np.quantile(bb, 0.025), 'hi_min': np.quantile(bb, 0.975), 'se_min': bb.std(ddof=1)})
    return out


if __name__ == '__main__':
    with Pool(6) as pool:
        resA = pool.map(part_a, OUTS)
        print('part A done', round(time.time() - t0), flush=True)
        resB = pool.map(part_b, KB, chunksize=2)
        print('part B done', round(time.time() - t0), flush=True)
    names = ['F5_disc_decomposition', 'F5_stage_dC', 'F5_capture_curves', 'F5_capture10_diff', 'F5_reclass_cells', 'F5_nri_ci', 'F5_dnb_ci']
    for j, nm in enumerate(names):
        pd.concat([r[j + 1] for r in resA]).to_csv(RV4 + nm + '.csv', index=False)
    B = pd.DataFrame(resB)
    meta = OM.set_index('key')
    B['disease'] = B.outcome.map(lambda x: meta.disease.get(x, x.replace('SYS_', '') + ' (system)'))
    B['system'] = B.outcome.map(lambda x: meta.system.get(x, 'System composite'))
    B['clock'] = B.outcome.map(lambda x: meta.clock.get(x, ''))
    B['set'] = np.where(B.outcome == 'NEG', 'Negative control', np.where(B.outcome.str.startswith('SYS_'), 'System composite', 'Tier 2'))
    for b_ in ['G', 'min']:
        B['p_' + b_] = 2 * norm.sf(np.abs(B['dC_' + b_] / B['se_' + b_]))
        m = B.set == 'Tier 2'; B['q_' + b_] = np.nan
        B.loc[m, 'q_' + b_] = multipletests(B.loc[m, 'p_' + b_], method='fdr_bh')[1]
    B.to_csv(RV4 + 'F5_tier2_dC_two_baselines.csv', index=False)
    # consistency checks against module 4
    M4 = pd.read_csv(R + 'M4_discrimination.csv').set_index('outcome')
    D_ = pd.read_csv(RV4 + 'F5_disc_decomposition.csv')
    g = D_[D_.model == 'G+gaps'].set_index('outcome')
    print(pd.DataFrame({'dC_new': g.dC, 'dC_M4': M4.dC_G, 'lo_new': g.lo, 'lo_M4': M4.dC_G_lo, 'hi_new': g.hi, 'hi_M4': M4.dC_G_hi}).round(4))
    T2 = pd.read_csv(R + 'M4_tier2_deltaC.csv').set_index('outcome')
    print('max |dC_min - M4 tier2|:', float((B.set_index('outcome').dC_min - T2.dC).abs().max()))
    print('total', round(time.time() - t0), flush=True)
