"""Figure 6 (v2): organ age along the guideline Calculate-Personalize-Reclassify (C-P-R) pathway.

Builds on fig6_v5_analysis (same cohort, exposures, outcomes; imported frame `s`).  New analyses:
 a  calibration of PREVENT-CVD by burden of aged target organs (PREVENT deciles x burden; AJ observed risk)
 b  threshold crossing and observational net benefit of upgrading people below the 5%, 7.5% and 20% action thresholds
 c  guideline risk enhancers: mutually adjusted HR, prevalence, population attributable fraction (Miettinen),
    cross-fitted dC of burden / 12 gaps beyond PREVENT + all available enhancers (complete case + missing indicator)
 d  hidden organ risk: AJ curves 0-12 y (heart->HF, kidney->CKD, brain->dementia) + biomarker/genetic sensitivity
 f  testing efficiency frontier in CKM stages 1-2 (continuous ranking by PREVENT / age, named strategies, bootstrap)
 S  subgroup consistency of the burden HR (Supplementary Fig. 18)
Outputs: results/v5_fig6v2/
"""
import sys, os, json, time
sys.path.insert(0, './analysis'); sys.path.insert(0, './code')
from fig6_v5_analysis import *          # builds `s`; helpers aj_cif, cif10, boot_cif, frame, hr; COV_A, COV_B, GAPS, ...
from scipy.stats import chi2
OUT2 = './results/v5_fig6v2/'; os.makedirs(OUT2, exist_ok=True)
t0 = time.time()
R = np.random.default_rng(20261018)


def save2(df, name):
    if isinstance(df, dict):
        json.dump(df, open(OUT2 + name + '.json', 'w'), indent=1, default=float)
    else:
        df.to_csv(OUT2 + name + '.csv', index=False)


def aj_curve(t, e, d, grid):
    t = np.asarray(t, float); e = np.asarray(e, float); d = np.asarray(d, float)
    ut, inv = np.unique(t, return_inverse=True)
    ne = np.bincount(inv, weights=e); nd = np.bincount(inv, weights=d); nn = np.bincount(inv)
    r = len(t) - np.r_[0, np.cumsum(nn)[:-1]]
    S = np.cumprod(1 - (ne + nd) / r); Sm = np.r_[1.0, S[:-1]]
    F = np.cumsum(Sm * ne / r)
    k = np.searchsorted(ut, grid, side='right') - 1
    return np.where(k >= 0, F[np.clip(k, 0, None)], 0.0)


def cif_boot_vec(g, key, nb, seed):
    """point estimate and bootstrap draws of 10-y AJ risk (proportion)."""
    t = g['t_' + key].values; e = g['ev_' + key].values
    dd = ((g['ev_' + key] == 0) & (g.died == 1)).astype(float).values
    est = aj_cif(t, e, dd)
    r = np.random.default_rng(seed); v = np.empty(nb)
    for b in range(nb):
        ix = r.integers(0, len(g), len(g)); v[b] = aj_cif(t[ix], e[ix], dd[ix])
    return est, v


# ------------------------------------------------------------------------------------------------ (a) calibration
d = frame(s, 'CVD').copy()
d['dec'] = pd.qcut(d.prevent_use.rank(method='first'), 10, labels=False)
rows = []
for q in range(10):
    for b in [0, 1, 2]:
        g = d[(d.dec == q) & (d.bcat == b)]
        est, v = cif_boot_vec(g, 'CVD', 200, 100 + q * 3 + b)
        rows.append(dict(decile=q + 1, bcat=b, n=len(g), events=int(g.ev_CVD.sum()), pred=g.prevent_use.mean(),
                         obs=100 * est, lo=100 * np.percentile(v, 2.5), hi=100 * np.percentile(v, 97.5)))
CAL = pd.DataFrame(rows); save2(CAL, 'F6a_calibration_deciles')
# overall O/E by burden with bootstrap CI (observed AJ / mean predicted), and calibration-in-the-large difference
rows = []
for b in [0, 1, 2]:
    g = d[d.bcat == b]; est, v = cif_boot_vec(g, 'CVD', 500, 200 + b); m = g.prevent_use.mean() / 100
    rows.append(dict(bcat=b, n=len(g), events=int(g.ev_CVD.sum()), obs=100 * est, pred=100 * m, oe=est / m,
                     oe_lo=np.percentile(v, 2.5) / m, oe_hi=np.percentile(v, 97.5) / m))
save2(pd.DataFrame(rows), 'F6a_oe_by_burden')
print('a', round(time.time() - t0)); print(pd.DataFrame(rows).round(3).to_string())

# ------------------------------------------------------------------------------------------------ (b) threshold crossing
# guideline anchors: 5% (upper bound of low risk), 7.5% (prioritisation of pharmacotherapy), 20% (stage-3 risk equivalent)
TIERS = [('<5%', 5.0), ('5-<7.5%', 7.5), ('7.5-<20%', 20.0)]
rows = []
for tier, pt in TIERS:
    g0 = d[d.prev_cat == tier]
    w = pt / 100 / (1 - pt / 100)
    for b in [0, 1, 2]:
        g = g0[g0.bcat == b]
        est, v = cif_boot_vec(g, 'CVD', 1000, 300 + int(pt * 10) + b)
        nb = 1000 * (est - (1 - est) * w); nbv = 1000 * (v - (1 - v) * w)
        share = len(g) / len(g0)
        rows.append(dict(tier=tier, threshold=pt, bcat=b, n=len(g), share_of_tier=100 * share, events=int(g.ev_CVD.sum()),
                         mean_prevent=g.prevent_use.mean(), obs=100 * est, lo=100 * np.percentile(v, 2.5), hi=100 * np.percentile(v, 97.5),
                         p_above=float((v > pt / 100).mean()),
                         dnb_per1000_upgraded=nb, dnb_lo=np.percentile(nbv, 2.5), dnb_hi=np.percentile(nbv, 97.5),
                         dnb_per1000_tier=nb * share, dnb_tier_lo=np.percentile(nbv, 2.5) * share, dnb_tier_hi=np.percentile(nbv, 97.5) * share))
TH = pd.DataFrame(rows); save2(TH, 'F6b_threshold_upgrade')
print(TH[['tier', 'bcat', 'n', 'share_of_tier', 'obs', 'lo', 'hi', 'p_above', 'dnb_per1000_upgraded', 'dnb_lo', 'dnb_hi', 'dnb_per1000_tier']].round(2).to_string())
# comparator: upgrading by a single guideline enhancer in the same tiers
s['enh_crp'] = (s.crp >= 2).astype(float)
s['enh_lpa'] = (s.lpa >= 125).astype(float).where(s.lpa.notna())
s['enh_uacr'] = (s.uacr >= 3).astype(float)
d = frame(s, 'CVD').copy()
rows = []
for tier, pt in TIERS:
    g0 = d[d.prev_cat == tier]; w = pt / 100 / (1 - pt / 100)
    for en, lab in [('b2', '>=2 aged organs'), ('enh_crp', 'hsCRP >=2 mg/L'), ('enh_lpa', 'Lp(a) >=125 nmol/L'), ('enh_uacr', 'UACR >=3 mg/mmol')]:
        g = g0[g0[en] == 1]
        est, v = cif_boot_vec(g, 'CVD', 500, 400 + int(pt * 10))
        rows.append(dict(tier=tier, threshold=pt, enhancer=lab, n=len(g), share_of_tier=100 * len(g) / len(g0), obs=100 * est,
                         lo=100 * np.percentile(v, 2.5), hi=100 * np.percentile(v, 97.5),
                         dnb_per1000_upgraded=1000 * (est - (1 - est) * w), dnb_lo=np.percentile(1000 * (v - (1 - v) * w), 2.5),
                         dnb_hi=np.percentile(1000 * (v - (1 - v) * w), 97.5)))
save2(pd.DataFrame(rows), 'F6b_threshold_upgrade_comparators')
print(pd.DataFrame(rows).round(2).to_string()); print('b', round(time.time() - t0))

# ------------------------------------------------------------------------------------------------ (c) enhancers: PAF + dC
s['enh_burden'] = s.b2
s['enh_apob'] = (s.apob >= 1.3).astype(float).where(s.apob.notna())
s['enh_tg'] = (s.tg >= 1.98).astype(float)
s['enh_ckd3a'] = ((s.egfr >= 45) & (s.egfr < 60)).astype(float)
s['enh_dep'] = (s.townsend >= s.townsend.quantile(0.8)).astype(float)
s['enh_depr'] = s.prior_depr; s['enh_inflam'] = s.prior_inflam
ENH = {'enh_burden': '>=2 aged organs', 'enh_crp': 'hsCRP >=2 mg/L', 'enh_lpa': 'Lp(a) >=125 nmol/L', 'enh_apob': 'ApoB >=1.3 g/L',
       'enh_tg': 'Triglycerides >=1.98 mmol/L', 'enh_uacr': 'UACR >=3 mg/mmol', 'enh_ckd3a': 'eGFR 45-59', 'enh_dep': 'Highest deprivation quintile',
       'enh_depr': 'Prior depression', 'enh_inflam': 'Chronic inflammatory disease'}
GROUP = {'enh_burden': 'organ age', 'enh_crp': 'biomarker', 'enh_lpa': 'biomarker', 'enh_apob': 'biomarker', 'enh_tg': 'biomarker',
         'enh_uacr': 'kidney', 'enh_ckd3a': 'kidney', 'enh_dep': 'social/clinical', 'enh_depr': 'social/clinical', 'enh_inflam': 'social/clinical'}
rows = []
for key in ['CVD', 'CKMC']:
    dd = frame(s, key).dropna(subset=list(ENH))
    r = hr(dd, key, list(ENH), ['log_prevent'] + COV_B)
    ev = dd['ev_' + key] == 1
    for i, (en, lab) in enumerate(ENH.items()):
        pc = dd.loc[ev, en].mean()
        paf = lambda h: 100 * pc * (h - 1) / h
        rows.append(dict(outcome=key, enhancer=lab, group=GROUP[en], hr=r.hr[i], lo=r.lo[i], hi=r.hi[i], p=r.p[i], prev=100 * dd[en].mean(),
                         prev_cases=100 * pc, paf=paf(r.hr[i]), paf_lo=paf(r.lo[i]), paf_hi=paf(r.hi[i]), n=len(dd), events=int(ev.sum())))
PAF = pd.DataFrame(rows); save2(PAF, 'F6c_enhancer_paf')
print(PAF[PAF.outcome == 'CVD'][['enhancer', 'hr', 'lo', 'hi', 'prev', 'paf', 'paf_lo', 'paf_hi']].round(2).to_string())


def crossfit_lp(dd, key, cols, reps=2, seed=500):
    lp = np.zeros(len(dd))
    for rep in range(reps):
        fo = np.random.default_rng(seed + rep).integers(0, 5, len(dd))
        for k in range(5):
            tr = dd[fo != k]; te = dd[fo == k]
            Xtr = tr[cols].values.astype(float)
            f = cox_fit(tr['t_' + key].values, tr['ev_' + key].values, Xtr)
            sd_ = Xtr.std(0); sd_[sd_ == 0] = 1
            lp[fo == k] += ((te[cols].values - f['mu']) / sd_) @ (f['beta'] * sd_) / reps
    return lp


rows = []
s['lpa_mi'] = s.lpa.isna().astype(float); s['apob_mi'] = s.apob.isna().astype(float)
for key in ['CVD', 'CKMC']:
    for samp in ['complete case', 'missing indicator']:
        if samp == 'complete case':
            dd = frame(s, key).dropna(subset=list(ENH)).reset_index(drop=True); enh = [e for e in ENH if e != 'enh_burden']
        else:
            dd = frame(s, key).copy().reset_index(drop=True)
            for c in ['enh_lpa', 'enh_apob']:
                dd[c] = dd[c].fillna(0)
            enh = [e for e in ENH if e != 'enh_burden'] + ['lpa_mi', 'apob_mi']
        base = ['log_prevent'] + COV_B + enh
        L = {'base': crossfit_lp(dd, key, base), '+burden': crossfit_lp(dd, key, base + ['b1', 'b2']), '+12 gaps': crossfit_lp(dd, key, base + GAPS)}
        t, e = dd['t_' + key].values, dd['ev_' + key].values
        cb = cindex(t, e, L['base'])
        for nm in ['+burden', '+12 gaps']:
            c1 = cindex(t, e, L[nm]); bs = []
            for _ in range(200):
                ix = R.integers(0, len(t), len(t)); bs.append(cindex(t[ix], e[ix], L[nm][ix]) - cindex(t[ix], e[ix], L['base'][ix]))
            rows.append(dict(outcome=key, sample=samp, model=nm, c_base=cb, c_new=c1, dc=c1 - cb, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             n=len(t), events=int(e.sum())))
        # likelihood ratio test for burden beyond all enhancers (full-sample fit)
        f0 = cox_fit(t, e, dd[base].values.astype(float)); f1 = cox_fit(t, e, dd[base + ['b1', 'b2']].values.astype(float))
        rows.append(dict(outcome=key, sample=samp, model='LR burden', c_base=np.nan, c_new=np.nan, dc=2 * (f1['ll'] - f0['ll']),
                         lo=np.nan, hi=chi2.sf(2 * (f1['ll'] - f0['ll']), 2), n=len(t), events=int(e.sum())))
DC = pd.DataFrame(rows); save2(DC, 'F6c_dc_beyond_enhancers'); print(DC.round(4).to_string()); print('c', round(time.time() - t0))

# ------------------------------------------------------------------------------------------------ (d) hidden organ risk curves
KID = [f'egfrcc_s{j}' for j in range(3)]
SPEC2 = [('Heart', 'HF', s.prev_hf < 5, ['nppb', 'tnni3_p'], '+ NT-proBNP + troponin I'),
         ('Kidney', 'CKD', (s.egfr >= 60) & ~(s.uacr >= 3), KID + ['cysc'], '+ eGFRcr-cys + cystatin C'),
         ('Brain', 'DEM', s.age >= 60, ['apoe4'], '+ APOE e4')]
grid = np.round(np.arange(0, 12.01, 0.1), 2)
rows = []; srows = []
for org, key, msk, extra, lab in SPEC2:
    g = frame(s[msk.values], key)
    for a in [0, 1]:
        h = g[g['aged_' + org] == a]
        t = h['t_' + key].values; e = h['ev_' + key].values; dd_ = ((h['ev_' + key] == 0) & (h.died == 1)).astype(float).values
        F = aj_curve(t, e, dd_, grid); B = np.empty((200, len(grid)))
        r_ = np.random.default_rng(600 + a)
        for b in range(200):
            ix = r_.integers(0, len(h), len(h)); B[b] = aj_curve(t[ix], e[ix], dd_[ix], grid)
        lo_, hi_ = np.percentile(B, [2.5, 97.5], axis=0)
        atrisk = [int((t >= x).sum()) for x in grid]
        for i, x in enumerate(grid):
            rows.append(dict(organ=org, outcome=key, aged=a, t=x, cif=100 * F[i], lo=100 * lo_[i], hi=100 * hi_[i], at_risk=atrisk[i]))
    for an, cv in [('main (model a + stage)', COV_B), (lab, COV_B + extra)]:
        r = hr(g, key, ['aged_' + org], cv)
        srows.append(dict(organ=org, outcome=key, adjustment=an, hr=r.hr[0], lo=r.lo[0], hi=r.hi[0], p=r.p[0], n=int(r.n[0]), events=int(r.events[0])))
CUR = pd.DataFrame(rows); save2(CUR, 'F6d_hidden_curves'); SEN = pd.DataFrame(srows); save2(SEN, 'F6d_hidden_sensitivity')
print(SEN.round(3).to_string()); print('d', round(time.time() - t0))

# ------------------------------------------------------------------------------------------------ (e) coordinated care: corrected dC
# v5 first version standardised the held-out design matrix twice (Z @ beta with beta on the original scale);
# here the cross-fitted linear predictor is (x - mu) @ beta (original scale), identical to the fitted Cox model.
rows = []
for key in ['OVL', 'MSYS', 'CVD']:
    dd = frame(s, key).reset_index(drop=True)
    base = COV_B + ['log_prevent']
    L = {'base': crossfit_lp(dd, key, base), '+burden': crossfit_lp(dd, key, base + ['b1', 'b2']), '+12 gaps': crossfit_lp(dd, key, base + GAPS)}
    t, e = dd['t_' + key].values, dd['ev_' + key].values
    cb = cindex(t, e, L['base'])
    for nm in ['+burden', '+12 gaps']:
        c1 = cindex(t, e, L[nm]); bs = []
        for _ in range(200):
            ix = R.integers(0, len(t), len(t)); bs.append(cindex(t[ix], e[ix], L[nm][ix]) - cindex(t[ix], e[ix], L['base'][ix]))
        rows.append(dict(outcome=key, model=nm, c_base=cb, c_new=c1, dc=c1 - cb, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         n=len(t), events=int(e.sum())))
DCe = pd.DataFrame(rows); save2(DCe, 'F6e_dc_corrected'); print(DCe.round(4).to_string()); print('e', round(time.time() - t0))

# ------------------------------------------------------------------------------------------------ (f) testing efficiency frontier
d12 = s[s.stage_use.isin([1, 2])].copy().reset_index(drop=True)
yany = ((d12.ev_CKMC == 1) & (d12.t_CKMC <= 10)).values
pos = (d12.b2 == 1).values
upc = pos & (d12.prevent_use < 7.5).values                    # would move up to intermediate-risk management
tgt_ev = pos & yany
n12 = len(d12)
fr = np.linspace(0, 1, 101)
rows = []
for nm, score in [('Rank by PREVENT', d12.prevent_use.values), ('Rank by age', d12.age.values)]:
    o = np.argsort(-score + 1e-9 * R.random(n12), kind='mergesort')
    ce = np.r_[0, np.cumsum(tgt_ev[o])]; cu = np.r_[0, np.cumsum(upc[o])]; cp = np.r_[0, np.cumsum(pos[o])]
    for f_ in fr:
        k = int(round(f_ * n12))
        rows.append(dict(strategy=nm, pct_tested=100 * f_, pct_events=100 * ce[k] / tgt_ev.sum(), pct_upclass=100 * cu[k] / upc.sum(),
                         flagged_per1000=1000 * cp[k] / max(k, 1)))
FR = pd.DataFrame(rows); save2(FR, 'F6f_frontier_curves')
STR = [('Test all', np.ones(n12, bool)),
       ('PREVENT 5-<20%', ((d12.prevent_use >= 5) & (d12.prevent_use < 20)).values),
       ('PREVENT >=7.5%', (d12.prevent_use >= 7.5).values),
       ('Age >=60 years', (d12.age >= 60).values),
       ('PREVENT 5-<20% or age >=60', (((d12.prevent_use >= 5) & (d12.prevent_use < 20)) | (d12.age >= 60)).values),
       ('Stage 2 only', (d12.stage_use == 2).values)]
rows = []
for nm, m in STR:
    def stat(ix):
        mm = m[ix]; te = tgt_ev[ix]; uu = upc[ix]
        return (100 * mm.mean(), 100 * (mm & te).sum() / te.sum(), 100 * (mm & uu).sum() / max(uu.sum(), 1), mm.sum() / max((mm & te).sum(), 1),
                100 * (mm & te).sum() / max((mm & pos[ix]).sum(), 1))
    est = stat(np.arange(n12)); B = np.array([stat(R.integers(0, n12, n12)) for _ in range(300)])
    lo_, hi_ = np.percentile(B, [2.5, 97.5], axis=0)
    rows.append(dict(strategy=nm, pct_tested=est[0], pct_events=est[1], pct_events_lo=lo_[1], pct_events_hi=hi_[1], pct_upclass=est[2],
                     pct_upclass_lo=lo_[2], pct_upclass_hi=hi_[2], tested_per_event=est[3], tpe_lo=lo_[3], tpe_hi=hi_[3], ppv_flagged=est[4],
                     n_tested=int(m.sum())))
NS = pd.DataFrame(rows); save2(NS, 'F6f_named_strategies')
save2(dict(n_stage12=n12, n_flagged=int(pos.sum()), n_target_events=int(tgt_ev.sum()), n_upclass=int(upc.sum()),
           n_prevent_ge20=int((d12.prevent_use >= 20).sum())), 'F6f_counts')
print(NS.round(2).to_string()); print('f', round(time.time() - t0))

print('fig6v2 done', round(time.time() - t0))
