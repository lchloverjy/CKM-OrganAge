"""Supplementary Fig. 18 (supports Fig. 6): consistency of the burden HR across prespecified subgroups."""
import sys, os, json, time
sys.path.insert(0, './analysis'); sys.path.insert(0, './code')
from fig6_v5_analysis import *
from scipy.stats import chi2
OUT2 = './results/v5_fig6v2/'; t0 = time.time()
def save2(df, name):
    df.to_csv(OUT2 + name + '.csv', index=False)
s['apoe4c'] = (s.apoe4 >= 1).astype(float)
# ------------------------------------------------------------------------------------------------ (S) subgroup consistency
SUB = [('Sex', 'male', [(0, 'Women'), (1, 'Men')]),
       ('Age', 'age60', [(0, '<60 years'), (1, '>=60 years')]),
       ('CKM stage', 'sg', [(0, 'Stage 0-1'), (1, 'Stage 2'), (2, 'Stage 3')]),
       ('PREVENT-CVD', 'prev75', [(0, '<7.5%'), (1, '>=7.5%')]),
       ('Diabetes', 'dm_bv', [(0, 'No'), (1, 'Yes')]),
       ('Statin use', 'statin', [(0, 'No'), (1, 'Yes')]),
       ('Ethnicity', 'white', [(1, 'White'), (0, 'Other')]),
       ('APOE e4', 'apoe4c', [(0, 'Non-carrier'), (1, 'Carrier')])]
s['age60'] = (s.age >= 60).astype(float); s['prev75'] = (s.prevent_use >= 7.5).astype(float)
rows = []
for key in ['CVD', 'CKMC']:
    dk = frame(s, key)
    for fac, var, levels in SUB:
        if min(int(dk.loc[dk[var] == lv, 'ev_' + key].sum()) for lv, _ in levels) < 30:
            continue
        cov = [c for c in ['log_prevent'] + COV_B if c != var and not (var == 'sg' and c in STG)]
        for lv, lab in levels:
            g = dk[dk[var] == lv]
            print(key, fac, lab, len(g), int(g['ev_' + key].sum()), flush=True)
            cv = [c for c in cov if g[c].std() > 0]
            r = hr(g, key, ['b1', 'b2'], cv)
            rows.append(dict(outcome=key, factor=fac, level=lab, n=int(r.n[1]), events=int(r.events[1]), hr=r.hr[1], lo=r.lo[1], hi=r.hi[1], p_int=np.nan))
        # interaction LR test (b1, b2 x subgroup indicators)
        dd = dk.dropna(subset=cov + [var]).copy()
        lv_ind = [f'{var}_{lv}' for lv, _ in levels[1:]]
        for lv, _ in levels[1:]:
            dd[f'{var}_{lv}'] = (dd[var] == lv).astype(float)
        inter = []
        for li in lv_ind:
            for b in ['b1', 'b2']:
                dd[f'{b}x{li}'] = dd[b] * dd[li]; inter.append(f'{b}x{li}')
        X0 = ['b1', 'b2'] + lv_ind + [c for c in cov if c not in lv_ind]
        X0 = [c for c in dict.fromkeys(X0) if dd[c].std() > 0]
        f0 = cox_fit(dd['t_' + key].values, dd['ev_' + key].values, dd[X0].values.astype(float))
        f1 = cox_fit(dd['t_' + key].values, dd['ev_' + key].values, dd[X0 + inter].values.astype(float))
        rows.append(dict(outcome=key, factor=fac, level='P interaction', n=len(dd), events=int(dd['ev_' + key].sum()), hr=np.nan, lo=np.nan, hi=np.nan,
                         p_int=chi2.sf(2 * (f1['ll'] - f0['ll']), len(inter))))
SG = pd.DataFrame(rows); save2(SG, 'SFig18_subgroups'); print(SG.round(3).to_string())
print('fig6v2 done', round(time.time() - t0))
