"""Numerically equivalent, faster Breslow Cox fit for the Fig. 3 bootstrap.

Same estimator, same Newton algorithm and convergence rule (|delta loglik| < 1e-9) as v3_common.cox_fit.
Speed-ups only: (1) the risk-set ordering (_prep) and the standardised covariate block are computed once per
outcome and reused for all 12 clocks; (2) cumulative sums are taken along contiguous rows of the transposed
design; (3) Newton iterations are warm-started from the previous clock's covariate coefficients.
Validated against v3_common.cox_fit (max |beta difference| reported by __main__)."""
import numpy as np


def prep_block(T, E, C):
    T = np.asarray(T, float); E = np.asarray(E, float); C = np.asarray(C, float)
    o = np.argsort(-T, kind='mergesort')
    Ts = T[o]
    _, first_idx, counts = np.unique(-Ts, return_index=True, return_counts=True)
    last = np.repeat(first_idx + counts - 1, counts)
    mu = C.mean(0); sd = C.std(0); sd[sd == 0] = 1
    Zc = ((C - mu) / sd)[o]
    Es = E[o]
    ev = Es == 1
    return {'o': o, 'last': last, 'ZcT': np.ascontiguousarray(Zc.T), 'sd_c': sd, 'ev': ev, 'last_ev': last[ev], 'n': len(T)}


def fit_exposure(P, x, b0=None, max_iter=40, tol=1e-9):
    """Cox fit of [x, C]; returns beta vector on the original scale (exposure first)."""
    x = np.asarray(x, float)
    mx = x.mean(); sx = x.std(); sx = sx if sx != 0 else 1.0
    zx = ((x - mx) / sx)[P['o']]
    ZT = np.vstack([zx[None, :], P['ZcT']])          # p x n, rows contiguous
    sd = np.concatenate([[sx], P['sd_c']])
    n = P['n']; p = ZT.shape[0]; ev = P['ev']; last = P['last']; lev = P['last_ev']
    b = np.zeros(p) if b0 is None else b0 * sd
    ll_old = -np.inf
    ZevT = ZT[:, ev]
    sumZev = ZevT.sum(1)
    for it in range(max_iter):
        eta = b @ ZT; m = eta.max(); w = np.exp(eta - m)
        S0 = np.cumsum(w)[last]
        S1T = np.cumsum(ZT * w, axis=1)[:, lev]       # p x n_events
        S0e = S0[ev]
        ll = np.sum(eta[ev] - m - np.log(S0e))
        Zb = (S1T / S0e).T                            # n_events x p
        g = sumZev - Zb.sum(0)
        acc = np.zeros(n); np.add.at(acc, lev, 1.0 / S0e)
        cj = np.cumsum(acc[::-1])[::-1]
        H = (ZT * (w * cj)) @ ZT.T - Zb.T @ Zb
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
    return b / sd


if __name__ == '__main__':
    import sys, time
    sys.path.insert(0, './analysis')
    import pandas as pd
    from v3_common import subcohort, cox_fit, COV_A, STG, ORG, ORG10, GAPS
    s = subcohort(); COVB = COV_A + STG
    OM = pd.read_csv('./results/V3_outcome_meta.csv')
    keys = OM[OM.clock.isin(ORG10)].key.tolist()
    rng = np.random.default_rng(1000); ix = rng.integers(0, len(s), len(s))
    db = s.iloc[ix].reset_index(drop=True)
    worst = 0; t_old = t_new = 0
    for k in keys[:4]:
        d = db[[f't_{k}', f'ev_{k}'] + GAPS + COVB].dropna()
        T, E, C = d[f't_{k}'].values, d[f'ev_{k}'].values, d[COVB].values
        t = time.time(); old = [cox_fit(T, E, np.column_stack([d['gap_' + o].values, C]))['beta'] for o in ORG]; t_old += time.time() - t
        t = time.time(); P = prep_block(T, E, C); prev = None; new = []
        for o in ORG:
            b0 = None if prev is None else np.concatenate([[0.0], prev[1:]])
            bb = fit_exposure(P, d['gap_' + o].values, b0); new.append(bb); prev = bb
        t_new += time.time() - t
        worst = max(worst, max(np.abs(a - b).max() for a, b in zip(old, new)))
    print(f'max |beta_old - beta_new| = {worst:.2e}; time old {t_old:.2f}s new {t_new:.2f}s; speed-up {t_old / t_new:.1f}x')
