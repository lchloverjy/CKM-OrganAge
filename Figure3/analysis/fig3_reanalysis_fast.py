"""Extended analyses for Fig. 3e,f (v5: bootstrap refits with code/fastcox.py, numerically identical to cox_fit).

(1) Organ correspondence across tier-2 diseases: for every disease with an organ-specific paired clock,
    excess = log HR(paired clock) - mean log HR(other nine organ clocks), model b.
    Clock-level correspondence index = mean excess over the diseases of that clock's organ system.
    95% CIs from 200 bootstrap samples of participants (all Cox models refitted in each sample).
(2) Kidney-function dependence: the same models with age gaps residualised on natural splines of
    eGFRcr-cys and log cystatin C (identical residualisation to the main kidney analysis).
"""
import sys, time, os
sys.path.insert(0, './code/v3')
from v3_common import *
from multiprocessing import Pool
sys.path.insert(0, './code')
from fastcox import prep_block, fit_exposure

OUT = './results/v4/'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
s = subcohort()
N = len(s)
COVB = COV_A + STG
OM = pd.read_csv('./results/V3_outcome_meta.csv')
PAIRS = OM[OM.clock.isin(ORG10)][['key', 'disease', 'system', 'clock', 'n_events']].reset_index(drop=True)

# kidney-function residualised gaps (as in m2.py)
Z = np.column_stack([ns_basis(s.egfr_cc.values), ns_basis(np.log(s.cysc.values)), np.ones(N)])
for o in ORG:
    y = s['gap_' + o].values
    bcoef = np.linalg.lstsq(Z, y, rcond=None)[0]
    r_ = y - Z @ bcoef
    s['kr_' + o] = (r_ - r_.mean()) / r_.std()

KEYS = PAIRS.key.tolist()
COLS = ['eid'] + [f't_{k}' for k in KEYS] + [f'ev_{k}' for k in KEYS] + GAPS + ['kr_' + o for o in ORG] + COVB
S = s[COLS].copy()


def fit_all(df, prefix):
    """beta for each disease x 12 clocks, model b (fastcox: same estimator as cox_fit, shared risk-set ordering)."""
    B = np.full((len(KEYS), len(ORG)), np.nan)
    for i, k in enumerate(KEYS):
        d = df[[f't_{k}', f'ev_{k}'] + [prefix + o for o in ORG] + COVB].dropna()
        P = prep_block(d[f't_{k}'].values, d[f'ev_{k}'].values, d[COVB].values)
        prev = None
        for j, o in enumerate(ORG):
            b0 = None if prev is None else np.concatenate([[0.0], prev[1:]])
            bb = fit_exposure(P, d[prefix + o].values, b0)
            B[i, j] = bb[0]; prev = bb
    return B
# ---------------------------------------------------------------- point estimates (raw and residualised, full HR table)
if os.path.exists(OUT + 'F3_tier2_raw_resid_hr.csv'):
    HR = pd.read_csv(OUT + 'F3_tier2_raw_resid_hr.csv')
else:
    rows = []
    for lab, pre in [('raw', 'gap_'), ('resid', 'kr_')]:
        for k in KEYS:
            for o in ORG:
                r, _ = cox_hr(S, f't_{k}', f'ev_{k}', [pre + o], COVB)
                r['outcome'] = k; r['clock'] = o; r['version'] = lab
                rows.append(r)
    HR = pd.concat(rows, ignore_index=True)
    HR.to_csv(OUT + 'F3_tier2_raw_resid_hr.csv', index=False)
print('point estimates done', round(time.time() - t0), 's', flush=True)


def excess_from_beta(B):
    """per-disease excess of paired clock over mean of other nine organ clocks"""
    ex = np.zeros(len(KEYS))
    for i, k in enumerate(KEYS):
        c = PAIRS.clock[i]
        b10 = B[i, :10]
        j = ORG10.index(c)
        ex[i] = b10[j] - np.delete(b10, j).mean()
    return ex


def boot_job(seed):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, N, N)
    db = S.iloc[ix].reset_index(drop=True)
    return excess_from_beta(fit_all(db, 'gap_')), excess_from_beta(fit_all(db, 'kr_'))


if __name__ == '__main__':
    Braw = HR[HR.version == 'raw'].pivot(index='outcome', columns='clock', values='beta').loc[KEYS, ORG].values
    Bres = HR[HR.version == 'resid'].pivot(index='outcome', columns='clock', values='beta').loc[KEYS, ORG].values
    ex_raw, ex_res = excess_from_beta(Braw), excess_from_beta(Bres)
    NB = 200
    with Pool(6) as P:
        out = P.map(boot_job, [1000 + b for b in range(NB)])
    EXR = np.array([o[0] for o in out]); EXS = np.array([o[1] for o in out])
    np.save(OUT + 'F3_boot_excess_raw.npy', EXR); np.save(OUT + 'F3_boot_excess_resid.npy', EXS)
    print('bootstrap done', round(time.time() - t0), 's', flush=True)
    # per-disease table
    D = PAIRS.copy()
    D['excess_raw'] = ex_raw; D['excess_resid'] = ex_res
    D['excess_raw_lo'] = np.percentile(EXR, 2.5, axis=0); D['excess_raw_hi'] = np.percentile(EXR, 97.5, axis=0)
    rank = []
    for i, k in enumerate(KEYS):
        b10 = Braw[i, :10]; j = ORG10.index(D.clock[i]); rank.append(int((b10 > b10[j]).sum() + 1))
    D['rank_raw'] = rank
    rank = []
    for i, k in enumerate(KEYS):
        b10 = Bres[i, :10]; j = ORG10.index(D.clock[i]); rank.append(int((b10 > b10[j]).sum() + 1))
    D['rank_resid'] = rank
    D.to_csv(OUT + 'F3_disease_correspondence.csv', index=False)
    # clock-level index (mean over own-system diseases)
    rows = []
    for c in ORG10:
        idx = np.where(D.clock.values == c)[0]
        if len(idx) == 0:
            continue
        for lab, ex, EX in [('raw', ex_raw, EXR), ('resid', ex_res, EXS)]:
            est = ex[idx].mean(); bs = EX[:, idx].mean(axis=1)
            se = bs.std(ddof=1)
            p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            from scipy.stats import norm
            rows.append({'clock': c, 'version': lab, 'n_diseases': len(idx), 'index': est,
                         'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5), 'se_boot': se,
                         'p_wald': 2 * norm.sf(abs(est / se)), 'p_boot': max(p, 1 / NB),
                         'n_rank1': int((D.loc[idx, 'rank_' + lab] == 1).sum())})
    CI = pd.DataFrame(rows)
    CI['q_bh'] = np.nan
    for lab in ['raw', 'resid']:
        m = CI.version == lab
        CI.loc[m, 'q_bh'] = bh(CI.loc[m, 'p_wald'].values)
    CI.to_csv(OUT + 'F3_clock_correspondence_index.csv', index=False)
    print(CI.round(4).to_string())
    print(D[['key', 'clock', 'excess_raw', 'excess_raw_lo', 'excess_raw_hi', 'excess_resid', 'rank_raw', 'rank_resid']].round(3).to_string())
    print('total', round(time.time() - t0), 's')
