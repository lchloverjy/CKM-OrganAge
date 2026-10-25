"""Figure 6 (v5): organ age at the decision points of the CKM care pathway.

All exposures, cohort, staging and outcomes are those of Figs 2-5 (full analytic cohort, baseline-verified stage,
burden of aged target organs, SD-scaled organ age gaps; outcomes from hospital inpatient + death-registry records,
censored at death or 30 Sep 2021).  Absolute risks: Aalen-Johansen with death as competing event.

a  Risk enhancer: 10-y total CVD by PREVENT category x number of aged target organs (0/1/>=2).
b  Benchmark against guideline risk-enhancing factors (HR adjusted for log PREVENT, stage, model a covariates).
c  Hidden organ risk: organ-specific disease among people judged normal by the routine organ assessment.
d  Coordinated care: >=2 newly affected organ systems and CKM-overlap by stage group x burden; cross-fitted dC.
e  Testing strategies in stages 1-2 (per 1,000 tested).
f  Imaging corroboration (CMR, aortic MRI, arterial stiffness, brain MRI) and subclinical damage index.
S  Full imaging atlas (82 CMR fields, 587 brain IDPs x 12 clocks) for the Supplement.
"""
import sys, os, json, time
sys.path.insert(0, './analysis'); sys.path.insert(0, './code')
import numpy as np, pandas as pd, statsmodels.api as sm
from importlib import import_module
from scipy.stats import norm
from v3_common import *
OUT = './results/v5_fig6/'; os.makedirs(OUT, exist_ok=True)
END = pd.Timestamp('2021-09-30')
rng = np.random.default_rng(2026)
t0 = time.time()


def save(df, name):
    if isinstance(df, dict):
        json.dump(df, open(OUT + name + '.json', 'w'), indent=1, default=float)
    else:
        df.to_csv(OUT + name + '.csv', index=False)


def aj_cif(t, e, d, horizon=10.0):
    t = np.asarray(t, float); e = np.asarray(e, float); d = np.asarray(d, float)
    ut, inv = np.unique(t, return_inverse=True)
    ne = np.bincount(inv, weights=e); nd = np.bincount(inv, weights=d); nn = np.bincount(inv)
    r = len(t) - np.r_[0, np.cumsum(nn)[:-1]]
    S = np.cumprod(1 - (ne + nd) / r); Sm = np.r_[1.0, S[:-1]]
    F = np.cumsum(Sm * ne / r)
    k = np.searchsorted(ut, horizon, side='right') - 1
    return float(F[k]) if k >= 0 else 0.0


def frame(d, key):
    return d[d['ev_' + key].notna() & d['t_' + key].notna()]


def cif10(d, key):
    if len(d) < 30:
        return np.nan
    dd = ((d['ev_' + key] == 0) & (d.died == 1)).astype(float).values
    return 100 * aj_cif(d['t_' + key].values, d['ev_' + key].values, dd)


def boot_cif(d, key, nb=200, seed=0):
    if len(d) < 30:
        return np.nan, np.nan
    r = np.random.default_rng(seed); t = d['t_' + key].values; e = d['ev_' + key].values
    dd = ((d['ev_' + key] == 0) & (d.died == 1)).astype(float).values
    v = []
    for _ in range(nb):
        ix = r.integers(0, len(d), len(d)); v.append(100 * aj_cif(t[ix], e[ix], dd[ix]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def hr(d, key, expo, covs):
    d = frame(d, key)
    r, _ = cox_hr(d, 't_' + key, 'ev_' + key, expo, covs)
    return r


# ------------------------------------------------------------------ cohort and derived variables
s = subcohort()
L1 = import_module('02_layer1'); L2 = import_module('03_layer2')
base = L1.load(); F = L2.extreme_flags(base); F['eid'] = base.eid.values
s = s.merge(F[['eid'] + TARGET].rename(columns={o: 'aged_' + o for o in TARGET}), on='eid', how='left')
s['bcat'] = np.minimum(s.burden_t, 2).astype(int); s['b1'] = (s.bcat == 1).astype(float); s['b2'] = (s.bcat == 2).astype(float)
s['sg'] = np.where(s.stage_use <= 1, 0, np.where(s.stage_use == 2, 1, 2))
COV_B = COV_A + STG
# PREVENT base equations (Khan 2024) for ASCVD and HF; total CVD = baseline-verified provider PREVENT (prevent_bv)
cf = pd.read_csv('/home/ubuntu/work/ext/prevent_base_10yr.csv')


def prevent(df, outc):
    ag = (df.age - 55) / 10; nh = (df.tc - df.hdl) - 3.5; hd = (df.hdl - 1.3) / 0.3
    s1 = (np.minimum(df.sbp, 110) - 110) / 20; s2 = (np.maximum(df.sbp, 110) - 130) / 20
    dm = df.dm_bv.astype(float); sm_ = (df.smoke == 2).astype(float)
    b1 = (np.minimum(df.bmi, 30) - 25) / 5; b2 = (np.maximum(df.bmi, 30) - 30) / 5
    e1 = (np.minimum(df.egfr, 60) - 60) / -15; e2 = (np.maximum(df.egfr, 60) - 90) / -15
    bp = df.htn_med.astype(float); st = df.statin.astype(float)
    T = [ag, nh, hd, s1, s2, dm, sm_, b1, b2, e1, e2, bp, st, bp * s2, st * nh, ag * nh, ag * hd, ag * s2, ag * dm, ag * sm_, ag * b2, ag * e1, 1.0]
    res = pd.Series(np.nan, index=df.index)
    for sx, lab in [(0, 'female'), (1, 'male')]:
        b = cf[f'{lab}_{outc}'].values; idx = df.male == sx
        lo_ = sum(b[i] * (T[i][idx] if not np.isscalar(T[i]) else T[i]) for i in range(23))
        res[idx] = 1 / (1 + np.exp(-lo_))
    return res


for c in ['tc', 'hdl']:
    s[c] = s[c].fillna(s[c].median())
s['prev_cvd_calc'] = 100 * prevent(s, 'total_cvd'); s['prev_ascvd'] = 100 * prevent(s, 'ascvd'); s['prev_hf'] = 100 * prevent(s, 'heart_failure')
r_chk = float(np.corrcoef(np.log(s.prev_cvd_calc), np.log(s.prevent_use.clip(0.1)))[0, 1])
s['prev_cat'] = pd.cut(s.prevent_use, [0, 5, 7.5, 20, 101], right=False, labels=['<5%', '5-<7.5%', '7.5-<20%', '>=20%']).astype(str)
# extra laboratory values (FIB-4) and pre-baseline HES conditions for guideline risk enhancers
X = pd.read_csv('/home/ubuntu/upload/data_all.csv', usecols=['Eid', 'AST', 'PLT']).rename(columns={'Eid': 'eid'})
s = s.merge(X, on='eid', how='left')
s['fib4'] = s.age * s.AST / (s.PLT * np.sqrt(s.alt.clip(1)))
Lh = pd.read_parquet('./data/dx/dx_hes_long.parquet')
Lh = Lh[Lh.eid.isin(s.eid)].merge(s[['eid', 'baseline']], on='eid'); Lh = Lh[Lh.date <= Lh.baseline]
def prior(prefixes):
    return s.eid.isin(set(Lh[Lh.code.str.match('^(' + '|'.join(prefixes) + ')')].eid)).astype(float)
s['prior_depr'] = prior(['F32', 'F33'])
s['prior_inflam'] = prior(['M05', 'M06', 'M32', 'M45', 'L40', 'K50', 'K51', 'B20', 'B21', 'B22', 'B23', 'B24'])
# composite total CVD (PREVENT definition: CHD, stroke, heart failure) from tier-2 components
comp = ['HF', 'CHD', 'ISTR', 'HSTR']
prev_any = np.zeros(len(s), bool); tmin = np.full(len(s), np.inf)
for k in comp:
    prev_any |= s['ev_' + k].isna().values
    tmin = np.minimum(tmin, np.where(s['ev_' + k] == 1, s['t_' + k], np.inf))
s['ev_CVD'] = np.where(prev_any, np.nan, np.isfinite(tmin).astype(float))
s['t_CVD'] = np.where(prev_any, np.nan, np.where(np.isfinite(tmin), tmin, s.fu_years))
# >=2 newly affected organ systems (multisystem disease) and CKM overlap (>=2 of T2D, CKD, CVD)
TS = np.column_stack([np.where(s['ev_SYS_' + k] == 1, s['t_SYS_' + k], np.inf) for k in SYSTEMS])
second = np.sort(TS, 1)[:, 1]
s['ev_MSYS'] = np.isfinite(second).astype(float); s['t_MSYS'] = np.where(np.isfinite(second), second, s.fu_years)
bl = np.column_stack([s.dm_bv.values == 1, s.egfr.values < 60, np.zeros(len(s), bool)])
tt = np.column_stack([np.where(s.ev_T2D == 1, s.t_T2D, np.inf), np.where(s.ev_CKD == 1, s.t_CKD, np.inf), np.where(s.ev_CVD == 1, s.t_CVD, np.inf)])
tt = np.where(bl, 0.0, tt); ov = np.sort(tt, 1)[:, 1]
elig = (bl.sum(1) <= 1) & s.ev_CVD.notna().values
s['ev_OVL'] = np.where(elig, (np.isfinite(ov) & (ov > 0)).astype(float), np.nan)
s['t_OVL'] = np.where(elig, np.where(np.isfinite(ov) & (ov > 0), ov, s.fu_years), np.nan)
facts = dict(n=len(s), r_prevent_calc_vs_provider=r_chk, n_cvd_events=int(s.ev_CVD.sum()), n_msys=int(s.ev_MSYS.sum()),
             n_ovl=int(np.nansum(s.ev_OVL)), n_ovl_at_risk=int(s.ev_OVL.notna().sum()), deaths=int(s.died.sum()),
             pct_b2=float(100 * s.b2.mean()), prev_cat=s.prev_cat.value_counts().to_dict(),
             fib4_obs=int(s.fib4.notna().sum()), uacr_obs=int(s.uacr.notna().sum()))
save(facts, 'fig6_facts'); print(facts)

if __name__ == '__main__':
    # -------------------------------------------------------------- (a) risk enhancer within PREVENT categories
    rows = []
    for key in ['CVD', 'CKMC']:
        d = frame(s, key)
        for pc in ['<5%', '5-<7.5%', '7.5-<20%', '>=20%']:
            for b in [0, 1, 2]:
                g = d[(d.prev_cat == pc) & (d.bcat == b)]
                lo, hi = boot_cif(g, key, 200, 11 + b)
                rows.append(dict(outcome=key, prev_cat=pc, bcat=b, n=len(g), events=int(g['ev_' + key].sum()), cif10=cif10(g, key),
                                 lo=lo, hi=hi, mean_prevent=g.prevent_use.mean()))
            g = d[d.prev_cat == pc]
            if g.b2.sum() > 30:
                r = hr(g, key, ['b1', 'b2'], ['log_prevent'] + COV_A)
                rows.append(dict(outcome=key, prev_cat=pc, bcat='HR_b2', n=len(g), events=int(r.events[1]), cif10=r.hr[1], lo=r.lo[1], hi=r.hi[1],
                                 mean_prevent=r.p[1]))
    A6 = pd.DataFrame(rows); save(A6, 'F6a_enhancer_cif')
    # overall burden HR beyond PREVENT, calibration (observed / PREVENT-predicted) by burden
    rows = []
    for key in ['CVD', 'CKMC']:
        r = hr(s, key, ['b1', 'b2'], ['log_prevent'] + COV_B); rows += [dict(outcome=key, term=t_, hr=r.hr[i], lo=r.lo[i], hi=r.hi[i], p=r.p[i]) for i, t_ in enumerate(['b1', 'b2'])]
        r = hr(s, key, ['burden_t'], ['log_prevent'] + COV_B); rows.append(dict(outcome=key, term='per_organ', hr=r.hr[0], lo=r.lo[0], hi=r.hi[0], p=r.p[0]))
    d = frame(s, 'CVD')
    for b in [0, 1, 2]:
        g = d[d.bcat == b]; lo, hi = boot_cif(g, 'CVD', 200, 21 + b)
        rows.append(dict(outcome='CVD', term=f'OE_b{b}', hr=cif10(g, 'CVD') / g.prevent_use.mean(), lo=lo / g.prevent_use.mean(), hi=hi / g.prevent_use.mean(), p=np.nan))
    save(pd.DataFrame(rows), 'F6a_overall')
    # people with >=2 aged organs whose observed risk crosses the next PREVENT threshold
    print('a', round(time.time() - t0))

    # -------------------------------------------------------------- (b) benchmark against guideline risk-enhancing factors
    s['enh_burden'] = s.b2
    s['enh_crp'] = (s.crp >= 2).astype(float)
    s['enh_lpa'] = (s.lpa >= 125).astype(float).where(s.lpa.notna())
    s['enh_apob'] = (s.apob >= 1.3).astype(float).where(s.apob.notna())
    s['enh_tg'] = (s.tg >= 1.98).astype(float)
    s['enh_ckd3a'] = ((s.egfr >= 45) & (s.egfr < 60)).astype(float)
    s['enh_uacr'] = (s.uacr >= 3).astype(float)          # unmeasured UACR treated as normal (as in KDIGO staging)
    s['enh_dep'] = (s.townsend >= s.townsend.quantile(0.8)).astype(float)
    s['enh_depr'] = s.prior_depr; s['enh_inflam'] = s.prior_inflam
    ENH = {'enh_burden': '>=2 aged organs', 'enh_crp': 'hsCRP >=2 mg/L', 'enh_lpa': 'Lp(a) >=125 nmol/L', 'enh_apob': 'ApoB >=1.3 g/L',
           'enh_tg': 'Triglycerides >=1.98 mmol/L', 'enh_uacr': 'UACR >=3 mg/mmol', 'enh_ckd3a': 'eGFR 45-59', 'enh_dep': 'Highest deprivation quintile',
           'enh_depr': 'Prior depression', 'enh_inflam': 'Chronic inflammatory disease'}
    rows = []
    for key in ['CVD', 'CKMC']:
        for en, lab in ENH.items():
            d = s[s[en].notna()]
            r = hr(d, key, [en], ['log_prevent'] + COV_B)
            rows.append(dict(outcome=key, enhancer=lab, model='single', hr=r.hr[0], lo=r.lo[0], hi=r.hi[0], p=r.p[0], prev=100 * d[en].mean(),
                             n=int(r.n[0]), events=int(r.events[0])))
        d = s.dropna(subset=list(ENH))
        r = hr(d, key, list(ENH), ['log_prevent'] + COV_B)
        for i, (en, lab) in enumerate(ENH.items()):
            rows.append(dict(outcome=key, enhancer=lab, model='mutual', hr=r.hr[i], lo=r.lo[i], hi=r.hi[i], p=r.p[i], prev=100 * d[en].mean(),
                             n=int(r.n[i]), events=int(r.events[i])))
    save(pd.DataFrame(rows), 'F6b_enhancers')
    print('b', round(time.time() - t0))

    # -------------------------------------------------------------- (c) hidden organ risk
    SPEC = [('Kidney', 'CKD', 'eGFR >=60 and UACR <3', (s.egfr >= 60) & ~(s.uacr >= 3)),
            ('Liver', 'CLD', 'FIB-4 <1.3', s.fib4 < 1.3),
            ('Pancreas', 'T2D', 'HbA1c <39 mmol/mol', (s.hba1c < 39) & (s.dm_bv == 0)),
            ('Heart', 'HF', 'PREVENT-HF <5%', s.prev_hf < 5),
            ('Artery', 'ASCVD', 'PREVENT-ASCVD <5%', s.prev_ascvd < 5),
            ('Brain', 'DEM', 'No routine assessment, age >=60', s.age >= 60)]
    rows = []
    for org, key, lab, msk in SPEC:
        d = frame(s[msk.values], key); ag = d['aged_' + org] == 1
        c1, c0 = cif10(d[ag], key), cif10(d[~ag], key)
        l1, h1 = boot_cif(d[ag], key, 200, 31); l0, h0 = boot_cif(d[~ag], key, 200, 32)
        r = hr(d, key, ['aged_' + org], COV_B)
        others = []
        for oc in TARGET:
            if oc != org:
                rr = hr(d, key, ['aged_' + oc], COV_B); others.append((rr.hr[0], oc))
        best = max(others)
        rows.append(dict(organ=org, outcome=key, normal_def=lab, n=len(d), pct_normal=100 * msk.mean(), pct_aged=100 * ag.mean(),
                         cif_aged=c1, cif_aged_lo=l1, cif_aged_hi=h1, cif_not=c0, cif_not_lo=l0, cif_not_hi=h0, rr=c1 / c0, rd=c1 - c0,
                         hr=r.hr[0], hr_lo=r.lo[0], hr_hi=r.hi[0], p=r.p[0], events=int(r.events[0]),
                         share_events_flagged=100 * d.loc[ag, 'ev_' + key].sum() / max(d['ev_' + key].sum(), 1),
                         best_other=best[1], best_other_hr=best[0]))
    C6 = pd.DataFrame(rows); C6['q'] = bh(C6.p.values); save(C6, 'F6c_hidden_organ_risk')
    print(C6[['organ', 'n', 'pct_aged', 'cif_aged', 'cif_not', 'hr', 'hr_lo', 'hr_hi', 'p']].round(3).to_string())
    print('c', round(time.time() - t0))

    # -------------------------------------------------------------- (d) coordinated care
    rows = []
    for key in ['MSYS', 'OVL']:
        d = frame(s, key)
        for sg in range(3):
            for b in [0, 1, 2]:
                g = d[(d.sg == sg) & (d.bcat == b)]; lo, hi = boot_cif(g, key, 200, 41 + sg * 3 + b)
                rows.append(dict(outcome=key, sg=sg, bcat=b, n=len(g), events=int(g['ev_' + key].sum()), cif10=cif10(g, key), lo=lo, hi=hi))
        r = hr(d, key, ['b1', 'b2'], ['log_prevent'] + COV_B)
        rows.append(dict(outcome=key, sg='all', bcat='HR_b2', n=int(r.n[1]), events=int(r.events[1]), cif10=r.hr[1], lo=r.lo[1], hi=r.hi[1]))
        r = hr(d, key, ['burden_t'], ['log_prevent'] + COV_B)
        rows.append(dict(outcome=key, sg='all', bcat='HR_per_organ', n=int(r.n[0]), events=int(r.events[0]), cif10=r.hr[0], lo=r.lo[0], hi=r.hi[0]))
    save(pd.DataFrame(rows), 'F6d_coordinated_care')
    # discrimination increment of the 12 gaps over stage + PREVENT + model a covariates (5-fold x 2 cross-fitting)
    rows = []
    for key in ['MSYS', 'OVL', 'CVD']:
        d = frame(s, key).reset_index(drop=True)
        base_cols = COV_B + ['log_prevent']; full = base_cols + GAPS
        res = {}
        for nm, cols in [('base', base_cols), ('base+gaps', full)]:
            lp = np.zeros(len(d))
            for rep in range(2):
                fo = np.random.default_rng(500 + rep).integers(0, 5, len(d))
                for k in range(5):
                    tr = d[fo != k]; te = d[fo == k]
                    f = cox_fit(tr['t_' + key].values, tr['ev_' + key].values, tr[cols].values)
                    Z = (te[cols].values - f['mu']) / np.where(tr[cols].values.std(0) == 0, 1, tr[cols].values.std(0))
                    lp[fo == k] += Z @ f['beta'] / 2
            res[nm] = lp
        t, e = d['t_' + key].values, d['ev_' + key].values
        cb, cf_ = cindex(t, e, res['base']), cindex(t, e, res['base+gaps'])
        bs = []
        for _ in range(100):
            ix = rng.integers(0, len(t), len(t)); bs.append(cindex(t[ix], e[ix], res['base+gaps'][ix]) - cindex(t[ix], e[ix], res['base'][ix]))
        rows.append(dict(outcome=key, c_base=cb, c_full=cf_, dc=cf_ - cb, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), events=int(e.sum()), n=len(t)))
    save(pd.DataFrame(rows), 'F6d_dc')
    print(pd.DataFrame(rows).round(4).to_string()); print('d', round(time.time() - t0))

    # -------------------------------------------------------------- (e) testing strategies in stages 1-2
    d = s[s.stage_use.isin([1, 2])].copy()
    def y10(key):
        return ((d['ev_' + key] == 1) & (d['t_' + key] <= 10)).astype(int).values
    yc = y10('CVD'); yk = y10('CKD'); yh = y10('HF'); ym = y10('DEM'); yl = y10('CLD'); yt = y10('T2D')
    yany = ((d.ev_CKMC == 1) & (d.t_CKMC <= 10)).astype(int).values
    STR = [('Test all stage 1-2', np.ones(len(d), bool)),
           ('PREVENT 5-<20%', ((d.prevent_use >= 5) & (d.prevent_use < 20)).values),
           ('Age >=60 years', (d.age >= 60).values),
           ('PREVENT 5-<20% or age >=60', (((d.prevent_use >= 5) & (d.prevent_use < 20)) | (d.age >= 60)).values)]
    rows = []
    tot_pos = (d.b2 == 1).values
    for nm, msk in STR:
        n_t = int(msk.sum()); per = 1000 / n_t
        pos = msk & tot_pos
        up = pos & (d.prevent_use < 7.5).values                          # moved up to intermediate-risk management
        hk = msk & (d.aged_Kidney == 1).values & (d.egfr >= 60).values & ~(d.uacr >= 3).values
        hh = msk & (d.aged_Heart == 1).values & (d.prev_hf < 5).values
        hl = msk & (d.aged_Liver == 1).values & (d.fib4 < 1.3).values
        ev_pos = int((pos & (yany == 1)).sum())
        rows.append(dict(strategy=nm, n_tested=n_t, pct_tested=100 * n_t / len(d), b2_per1000=pos.sum() * per, upclass_per1000=up.sum() * per,
                         ckm_events_in_b2_per1000=ev_pos * per, nnt_per_event=n_t / max(ev_pos, 1),
                         share_b2_events_captured=100 * ev_pos / max(int((tot_pos & (yany == 1)).sum()), 1),
                         ppv_b2=100 * ev_pos / max(pos.sum(), 1), ppv_not=100 * ((msk & ~tot_pos) & (yany == 1)).sum() / max((msk & ~tot_pos).sum(), 1),
                         hidden_kidney_per1000=hk.sum() * per, hidden_kidney_ckd_per1000=(hk & (yk == 1)).sum() * per,
                         hidden_heart_per1000=hh.sum() * per, hidden_heart_hf_per1000=(hh & (yh == 1)).sum() * per,
                         hidden_liver_per1000=hl.sum() * per, hidden_liver_cld_per1000=(hl & (yl == 1)).sum() * per))
    E6 = pd.DataFrame(rows); save(E6, 'F6e_testing_strategies'); print(E6.round(2).T.to_string())
    print('e', round(time.time() - t0))

    # -------------------------------------------------------------- (f) imaging corroboration
    s['asc_dist_'] = s.asc_dist; s['gls_'] = -s.gls.abs(); s['gcs_'] = -s.gcs.abs()
    PH = {'Heart': [('lvef', 'LVEF', -1), ('gls', 'LV GLS (magnitude)', -1), ('laef', 'LA EF', -1), ('lvm', 'LV mass', 1), ('concentric', 'LV concentricity', 1)],
          'Artery': [('asc_dist', 'Asc. aortic distensibility', -1), ('desc_dist', 'Desc. aortic distensibility', -1), ('asi', 'Arterial stiffness index', 1)],
          'Brain': [('tbv', 'Brain volume', -1), ('hippo', 'Hippocampal volume', -1), ('cort_thk', 'Cortical thickness', -1),
                    ('log_wmh', 'WM hypointensity volume (log)', 1), ('fa_genu', 'FA, genu of corpus callosum', -1)]}
    s['gls'] = s.gls.abs()
    rows = []
    for dom, lst in PH.items():
        for v, lab, sgn in lst:
            extra = ['bmi'] if dom in ['Heart', 'Artery'] else []
            d = s.dropna(subset=[v])
            lo_, hi_ = d[v].quantile([0.001, 0.999]); d = d[(d[v] >= lo_) & (d[v] <= hi_)]
            y = sgn * (d[v] - d[v].mean()) / d[v].std()
            for c in ORG:
                Xm = sm.add_constant(d[['gap_' + c] + COV_A + extra].astype(float)); fit = sm.OLS(y.values, Xm.values).fit(cov_type='HC3')
                b, se = fit.params[1], fit.bse[1]
                rows.append(dict(domain=dom, var=v, phenotype=lab, clock=c, beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=fit.pvalues[1], n=len(d)))
    IM = pd.DataFrame(rows); IM['q'] = bh(IM.p.values); save(IM, 'F6f_imaging')
    spec = []
    for dom in PH:
        m = IM[IM.domain == dom].groupby('clock').beta.mean().sort_values(ascending=False)
        spec.append(dict(domain=dom, top_clock=m.index[0], paired_rank=int(list(m.index).index(dom) + 1), paired_mean=m[dom], top_mean=m.iloc[0],
                         n_pheno=len(PH[dom]), n=int(IM[IM.domain == dom].n.max())))
    save(pd.DataFrame(spec), 'F6f_domain_rank'); print(pd.DataFrame(spec).round(3).to_string())
    # subclinical damage index (mean of oriented z within each organ domain) by burden category
    Z = {}
    for dom, lst in PH.items():
        zz = pd.concat([sgn * (s[v] - s[v].mean()) / s[v].std() for v, lab, sgn in lst], axis=1)
        Z[dom] = zz.mean(1, skipna=True).where(zz.notna().sum(1) >= 1)
    s['dmg_heart'] = Z['Heart']; s['dmg_artery'] = Z['Artery']; s['dmg_brain'] = Z['Brain']
    dm = pd.concat([Z['Heart'], Z['Artery'], Z['Brain']], axis=1)
    s['dmg_index'] = dm.mean(1).where(dm.notna().sum(1) >= 2)
    rows = []
    for col in ['dmg_index', 'dmg_heart', 'dmg_artery', 'dmg_brain']:
        d = s.dropna(subset=[col]); y = (d[col] - d[col].mean()) / d[col].std()
        fit = sm.OLS(y.values, sm.add_constant(d[['b1', 'b2'] + COV_A].astype(float)).values).fit(cov_type='HC3')
        ft = sm.OLS(y.values, sm.add_constant(d[['burden_t'] + COV_A].astype(float)).values).fit(cov_type='HC3')
        for b in [0, 1, 2]:
            est = 0.0 if b == 0 else fit.params[b]; se = 0.0 if b == 0 else fit.bse[b]
            rows.append(dict(index=col, bcat=b, diff=est, lo=est - 1.96 * se, hi=est + 1.96 * se, n=len(d), n_b=int((d.bcat == b).sum()),
                             per_organ=ft.params[1], per_organ_lo=ft.params[1] - 1.96 * ft.bse[1], per_organ_hi=ft.params[1] + 1.96 * ft.bse[1], p_trend=ft.pvalues[1]))
    DI = pd.DataFrame(rows); save(DI, 'F6f_damage_index'); print(DI.round(3).to_string())
    save(dict(n_cmr=int(s.lvef.notna().sum()), n_aorta=int(s.asc_dist.notna().sum()), n_asi=int(s.asi.notna().sum()),
              n_brain=int(s.tbv.notna().sum()), n_wmh=int(s.log_wmh.notna().sum()), n_dmg=int(s.dmg_index.notna().sum())), 'F6f_counts')
    print('f', round(time.time() - t0))

    # -------------------------------------------------------------- Supplement: full imaging atlas
    B = '/home/ubuntu/work/brain/原始脑区数据/'
    cm = pd.read_csv('/home/ubuntu/upload/cmr82.csv'); idc = cm.columns[0]
    cmr = cm[[c for c in cm.columns if c.endswith('-2.0')]].copy(); cmr.columns = [c.split('-')[0] for c in cmr.columns]; cmr['eid'] = cm[idc].values
    br = pd.read_csv(B + 'data_587idp_v2.csv').rename(columns={'Eid': 'eid'})
    tb = pd.read_csv(B + 'total_brain_volumes.csv').rename(columns={'Eid': 'eid'})[['eid', 'Scaling']]
    rows = []
    for src, tab, extra in [('CMR', cmr, ['bmi']), ('Brain', br.merge(tb, on='eid', how='left'), ['Scaling'])]:
        dd = s[['eid'] + GAPS + COV_A + ['bmi']].merge(tab, on='eid', how='inner')
        cols = [c for c in tab.columns if c not in ('eid', 'Scaling')]
        Xb = dd[COV_A + extra].astype(float)
        for ph in cols:
            v = pd.to_numeric(dd[ph], errors='coerce'); m = v.notna() & Xb.notna().all(1)
            if m.sum() < 500 or v[m].std() == 0:
                continue
            lo_, hi_ = v[m].quantile([0.001, 0.999]); m = m & (v >= lo_) & (v <= hi_)
            y = ((v[m] - v[m].mean()) / v[m].std()).values; Xc = np.c_[np.ones(m.sum()), Xb[m].values]
            bcv, *_ = np.linalg.lstsq(Xc, y, rcond=None); ry = y - Xc @ bcv
            for c in ORG:
                z = dd.loc[m, 'gap_' + c].values; bz, *_ = np.linalg.lstsq(Xc, z, rcond=None); rz = z - Xc @ bz
                beta = (rz @ ry) / (rz @ rz); res_ = ry - beta * rz; se = np.sqrt(res_ @ res_ / (len(ry) - Xc.shape[1] - 1) / (rz @ rz))
                rows.append(dict(source=src, field=ph, clock=c, beta=beta, se=se, p=2 * norm.sf(abs(beta / se)), n=int(m.sum())))
    AT = pd.DataFrame(rows); AT['q'] = AT.groupby('source').p.transform(lambda x: bh(x.values)); save(AT, 'SFig_imaging_atlas')
    print('atlas', AT.groupby(['source', 'clock']).apply(lambda g: (g.q < 0.05).sum()).unstack(0).to_string())
    print('fig6 done', round(time.time() - t0))
