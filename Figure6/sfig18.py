"""Supplementary Fig. 18. Consistency of the association of >=2 vs 0 aged target organs across prespecified subgroups.
Reads ../source_data/SFig18_subgroups.csv.   Run: cd code && python sfig18.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style4 import *
import pandas as pd, numpy as np
SD = os.path.join(HERE, '..', 'source_data') + '/'
S = pd.read_csv(SD + 'SFig18_subgroups.csv')
FIX = lambda t: str(t).replace('>=', '≥').replace('0-1', '0–1').replace('APOE e4', 'APOE ε4')
S['level'] = S.level.map(FIX); S['factor'] = S.factor.map(FIX)
OUTC = [('CVD', 'Total CVD', '#A4161A'), ('CKMC', 'CKM composite', '#2F5D8C')]


def panel(C, key, ttl, col, x0):
    g = S[S.outcome == key]
    facs = list(dict.fromkeys(g.factor))
    rows = []
    for f in facs:
        gg = g[g.factor == f]
        pint = gg[gg.level == 'P interaction'].p_int.values
        rows.append(('h', f, pint[0] if len(pint) else np.nan, None))
        for _, r in gg[gg.level != 'P interaction'].iterrows():
            rows.append(('r', r.level, r, None))
    rh = 3.05; y0 = 13.0
    ax = C.ax(x0 + 36, y0, 30, rh * len(rows))
    ax.set_xscale('log'); ax.set_xlim(0.9, 3.4); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axvline(1, color=INK, lw=0.45)
    allr = g[g.level == 'Overall']
    C.text(x0 + 4.3, 3.2, ttl, fontsize=6.5, fontweight='bold', va='top', color=col)
    C.text(x0 + 4.3, 8.6, 'Subgroup', fontsize=5.3, color=MID)
    C.text(x0 + 30.0, 8.6, 'n / events', fontsize=5.3, color=MID, ha='right')
    C.text(x0 + 51, 8.6, 'HR (95% CI), ≥2 vs 0 aged organs', fontsize=5.3, color=MID, ha='center')
    C.text(x0 + 76.5, 8.6, 'P int.', fontsize=5.3, color=MID, ha='center')
    for i, (kind, lab, r, _) in enumerate(rows):
        yy = y0 + rh * (i + 0.5)
        if kind == 'h':
            C.text(x0 + 4.3, yy, lab, fontsize=5.6, fontweight='bold', va='center')
            if not np.isnan(r):
                C.text(x0 + 76.5, yy, fmt_p(r) if r < 0.001 else f'{r:.2f}', fontsize=5.4, ha='center', va='center',
                       color=INK if r < 0.05 else MID)
        else:
            C.text(x0 + 6.5, yy, lab, fontsize=5.4, va='center')
            C.text(x0 + 30.0, yy, f'{int(r.n):,} / {int(r.events):,}', fontsize=5.2, ha='right', va='center', color=MID)
            ci(ax, np.array([r.hr]), np.array([i]), np.array([r.lo]), np.array([r.hi]), color=col, ms=2.4, lw=0.6)
            C.text(x0 + 68.5, yy, um(f'{r.hr:.2f} ({r.lo:.2f}–{r.hi:.2f})'), fontsize=5.2, ha='left', va='center')
    ax.set_yticks([]); ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(['1', '1.5', '2', '3']); ax.minorticks_off()
    clean(ax, left=False); ax.set_xlabel('Hazard ratio (log scale)')
    return y0 + rh * len(rows)


def full(C):
    for k, (key, ttl, col) in enumerate(OUTC):
        panel(C, key, ttl, col, k * 92)
    C.text(4.3, 104.5, 'Cox models adjusted for log PREVENT-CVD, CKM stage and model a covariates (the stratifying variable omitted). Diabetes is not shown for the CKM composite',
           fontsize=5.2, color=MID)
    C.text(4.3, 107.0, 'because prevalent diabetes is a component. Nominal interaction P values are unadjusted for ~15 tests; none would remain significant after Bonferroni correction.',
           fontsize=5.2, color=MID)


if __name__ == '__main__':
    single(full, 184, 110, 'Supplementary_Fig18_subgroups', os.path.join(HERE, '..', 'supplementary'))
    print('done')
