"""Figure 2. Organ age gaps across and within CKM stages.
a, stage trajectories of the 12 age gaps (M2), with kidney-function-residualised trajectories and onset shares;
b, variance in age gaps explained by six CKM-related trait domains (Shapley decomposition);
c, stage differences before (M2) and after (M3) adjustment for the stage-defining traits;
d, contributions of the stage-defining trait domains to the change from M2 to M3 (Gelbach decomposition);
e, overlap of age gaps across stages; f, multi-organ ageing burden; g, identification of high burden in stages 1-2.
Reads only ../source_data (exported by prep_fig2.py).  Run:  python fig2.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style4 import (Canvas, MM, _save, letter, clean, ci, um, fmt_p, lighten, INK, MID, LIGHT, FAINT,
                    STAGE_COL, STAGE_LAB, ORGAN_COL, ACCENT, BLUE)
SD = os.path.join(HERE, '..', 'source_data') + os.sep
ROOT = os.path.join(HERE, '..')
M2 = pd.read_csv(SD + 'Fig2a_stage_differences_M2.csv')
KR = pd.read_csv(SD + 'Fig2a_kidney_residualised.csv')
ON = pd.read_csv(SD + 'Fig2a_onset_share.csv').set_index('clock')
AS = pd.read_csv(SD + 'Fig2a_asynchrony_test.csv').iloc[0]
SH = pd.read_csv(SD + 'Fig2b_shapley.csv')
MM3 = pd.read_csv(SD + 'Fig2c_stage_differences_M2_M3.csv')
GB = pd.read_csv(SD + 'Fig2d_gelbach_domains.csv')
WS = pd.read_csv(SD + 'Fig2e_within_stage.csv').set_index('clock')
DEN = pd.read_csv(SD + 'Fig2e_density.csv')
BD = pd.read_csv(SD + 'Fig2f_burden_distribution.csv')
OR = pd.read_csv(SD + 'Fig2f_odds_high_burden.csv')
BR = pd.read_csv(SD + 'Fig2f_burden_kidney_residualised.csv')
AUC = pd.read_csv(SD + 'Fig2g_auc.csv').set_index('model')
ROC = pd.read_csv(SD + 'Fig2g_roc.csv')
SMD = pd.read_csv(SD + 'Fig2g_smd.csv')
PC = pd.read_csv(SD + 'Fig2g_prevent_categories.csv')

DOMS = ['Kidney function', 'Adiposity', 'Glycaemia', 'Blood pressure', 'Lipids', 'Inflammation']
DCOL = {'Kidney function': '#2F5D8C', 'Adiposity': '#C9A227', 'Glycaemia': '#D98C6A',
        'Blood pressure': '#A85A6A', 'Lipids': '#8C8C8C', 'Inflammation': '#CFCFCF'}
LATE_BAND = '#F4F1F6'
LATE_INK = '#6D597A'
# common clock order: data-driven early-rising group, then late-rising group, each by decreasing stage 3 difference
_s3 = M2[M2.stage == 3].set_index('clock').beta_sd
_e = _s3[ON.loc[_s3.index, 'group'] == 'early'].sort_values(ascending=False).index.tolist()
_l = _s3[ON.loc[_s3.index, 'group'] == 'late'].sort_values(ascending=False).index.tolist()
ROWS = _e + _l
NE = len(_e)
LATE = set(_l)


# ---------------------------------------------------------------- a  stage trajectories
def panel_a(C):
    x0, w, g, h = 10.0, 16.2, 1.75, 23.5
    ys = [10.0, 42.0]
    grid = [ROWS[:6], ROWS[6:]]
    # legend
    C.line([10.0, 13.0], [2.6, 2.6], color=INK, lw=1.0)
    C.rect(10.0, 1.9, 3.0, 1.4, fc=lighten(INK, 0.8), ec='none', zorder=0)
    C.text(13.8, 2.6, 'M2, 95% CI', fontsize=5.2, va='center')
    C.line([29.0, 32.0], [2.6, 2.6], color=MID, lw=0.75, ls=(0, (2.2, 1.2)))
    C.text(32.8, 2.6, 'After residualising on eGFR and cystatin C', fontsize=5.2, va='center')
    C.text(77.0, 2.6, 'Filled point, FDR q < 0.05', fontsize=5.2, va='center', color=MID)
    C.text(10.0, 6.0, um(f'Percentage under each name: stage 1 difference as a share of the stage 3 difference; '
                         f'equal shares across clocks, P = ') + fmt_p(AS.p), fontsize=5.0, va='center', color=MID)
    for r, row in enumerate(grid):
        for j, cl in enumerate(row):
            ax = C.ax(x0 + j * (w + g), ys[r], w, h)
            d = M2[M2.clock == cl].sort_values('stage')
            k = KR[KR.clock == cl].sort_values('stage')
            x = np.arange(4)
            b = np.r_[0, d.beta_sd.values]
            lo = np.r_[0, d.lo_sd.values]
            hi = np.r_[0, d.hi_sd.values]
            col = ORGAN_COL[cl]
            if cl in LATE:
                ax.set_facecolor(LATE_BAND)
            ax.axhline(0, color=LIGHT, lw=0.4, ls=(0, (2, 1.5)), zorder=0)
            ax.plot(x, np.r_[0, k.beta_sd.values], color=MID, lw=0.75, ls=(0, (2.2, 1.2)), zorder=2)
            ax.fill_between(x, lo, hi, color=lighten(col, 0.65), lw=0, zorder=1)
            ax.plot(x, b, color=col, lw=1.0, zorder=3)
            sig = np.r_[True, d.q.values < 0.05]
            ax.scatter(x, b, s=7, fc=[col if s_ else 'white' for s_ in sig], ec=col, lw=0.6, zorder=4)
            ax.text(3.18, b[3], f'{b[3]:.2f}', fontsize=5.1, color=col, va='center', ha='left')
            ax.set_ylim(-0.3, 0.98)
            ax.set_xlim(-0.3, 4.0)
            ax.set_xticks(x)
            ax.set_xticklabels(['0', '1', '2', '3'] if r == 1 else [])
            ax.set_yticks([0, 0.4, 0.8])
            if j:
                ax.set_yticklabels([])
            ax.text(0.05, 0.97, cl, transform=ax.transAxes, fontsize=5.8, fontweight='bold', color=col,
                    va='top', ha='left')
            ax.text(0.05, 0.80, um(f'{100 * ON.loc[cl, "share1"]:.0f}%'), transform=ax.transAxes, fontsize=5.0,
                    color=MID, va='top', ha='left')
            clean(ax)
    C.text(2.6, ys[0] + h + (ys[1] - ys[0] - h) / 2, 'Difference in age gap vs stage 0 (s.d.)', rotation=90,
           fontsize=6.0, va='center', ha='center')
    C.text(x0 + 3 * (w + g) - g / 2, ys[1] + h + 5.8, 'CKM stage', fontsize=6.0, ha='center')
    # late-rising cluster bracket
    xl = x0 + 3 * (w + g)
    xr = xl + 3 * w + 2 * g
    C.line([xl, xr], [ys[1] - 1.6, ys[1] - 1.6], lw=0.6, color=LATE_INK)
    C.text((xl + xr) / 2, ys[1] - 2.1, um(f'Late-rising cluster (in {100 * AS.late_group_reproduced_in_bootstrap:.1f}% '
                                         f'of {int(AS.bootstrap_samples)} bootstrap samples)'),
           fontsize=5.0, color=LATE_INK, ha='center', va='bottom')


# ---------------------------------------------------------------- b  Shapley decomposition
def panel_b(C):
    tot = SH.groupby('clock').total_r2.first().sort_values()
    for k, c in enumerate(DOMS):
        xx = 3.0 + (k % 3) * 19.0
        yy = 2.6 + (k // 3) * 3.4
        C.rect(xx, yy - 0.85, 2.0, 1.7, fc=DCOL[c], ec='none')
        C.text(xx + 2.7, yy, c, fontsize=5.2, va='center')
    ax = C.ax(18.5, 12.0, 34.0, 50.5)
    clean(ax, left=False, grid='x')
    for i, cl in enumerate(tot.index):
        d = SH[SH.clock == cl].set_index('component')
        left = 0
        for c in DOMS:
            v = d.loc[c, 'r2'] * 100 if c in d.index else 0
            ax.barh(i, v, left=left, height=0.68, color=DCOL[c], lw=0)
            left += v
        ax.text(left + 0.25, i, f'{tot[cl] * 100:.1f}', fontsize=5.1, va='center', color=MID)
    ax.set_yticks(range(len(tot)))
    ax.set_yticklabels(tot.index, fontsize=5.6)
    for t, cl in zip(ax.get_yticklabels(), tot.index):
        t.set_color(ORGAN_COL[cl])
    ax.set_xlim(0, 15.0)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-0.6, len(tot) - 0.4)
    ax.set_xlabel('Variance in age gap explained\nbeyond age and sex (%)', fontsize=5.8, labelpad=1.2,
                  linespacing=1.0)


# ---------------------------------------------------------------- c, d  shared clock grid
Y0, RH, GAP = 12.0, 3.3, 2.0


def ry(i):
    return Y0 + i * RH + (GAP if i >= NE else 0) + RH / 2


YB = Y0 + len(ROWS) * RH + GAP


def late_band(C, x0, x1):
    C.rect(x0, ry(NE) - RH / 2 - 0.5, x1 - x0, len(_l) * RH + 1.0, fc=LATE_BAND, ec='none', lw=0, zorder=0)


def row_labels(C, x):
    for i, cl in enumerate(ROWS):
        C.text(x, ry(i), cl, fontsize=5.6, ha='right', va='center', color=ORGAN_COL[cl])


def grid_axes(C, x, w):
    ax = C.ax(x, Y0, w, YB - Y0)
    ax.set_ylim(YB, Y0)
    ax.set_yticks([])
    ax.patch.set_alpha(0)
    return ax


def panel_c(C):
    late_band(C, 1.0, 90.5)
    row_labels(C, 15.8)
    C.text(2.2, ry(NE + 1), 'Late rising', fontsize=5.0, rotation=90, ha='center', va='center', color=LATE_INK)
    C.line([17.5], [3.0], marker='o', ms=2.6, color=INK, lw=0)
    C.text(18.9, 3.0, 'M2', fontsize=5.3, va='center')
    C.line([25.5], [3.0], marker='o', ms=2.6, mfc='white', mec=INK, mew=0.6, lw=0)
    C.text(26.9, 3.0, 'M3 (+ stage-defining traits)', fontsize=5.3, va='center')
    C.line([55.5], [3.0], marker='o', ms=2.6, mfc='white', mec=LIGHT, mew=0.6, lw=0)
    C.text(56.9, 3.0, 'M3, 95% CI includes 0', fontsize=5.3, va='center', color=MID)
    lo, hi = -0.36, 0.92
    for k, s in enumerate([1, 2, 3]):
        x = 17.0 + k * 25.0
        ax = grid_axes(C, x, 23.0)
        clean(ax, left=False, grid='x')
        ax.axvline(0, color=MID, lw=0.45, zorder=1)
        for i, cl in enumerate(ROWS):
            a = MM3[(MM3.clock == cl) & (MM3.model == 'M2') & (MM3.stage == s)].iloc[0]
            b = MM3[(MM3.clock == cl) & (MM3.model == 'M3') & (MM3.stage == s)].iloc[0]
            col = ORGAN_COL[cl]
            keep = (b.lo_sd > 0) or (b.hi_sd < 0)
            ax.plot([a.beta_sd, b.beta_sd], [ry(i), ry(i)], color=lighten(col, 0.45), lw=0.95, zorder=2)
            ax.scatter(a.beta_sd, ry(i), s=8.5, fc=col, ec=col, lw=0.5, zorder=4)
            ax.scatter(b.beta_sd, ry(i), s=8.5, fc='white', ec=col if keep else LIGHT, lw=0.7, zorder=4)
            if keep and b.beta_sd > 0.15:
                ax.text(b.beta_sd - 0.06, ry(i), f'{b.beta_sd:.2f}', fontsize=4.9, ha='right', va='center',
                        color=col, fontweight='bold')
        ax.set_xlim(lo, hi)
        ax.set_xticks([-0.3, 0, 0.3, 0.6])
        ax.set_xticklabels([um('-0.3'), '0', '0.3', '0.6'])
        C.text(x + 23.0 / 2, Y0 - 1.0, f'Stage {s} vs 0', fontsize=5.8, ha='center', va='bottom',
               color=STAGE_COL[s], fontweight='bold')
    C.text(17.0 + 36.5, YB + 4.6, 'Difference in age gap vs stage 0 (s.d.)', fontsize=5.8, ha='center', va='top')


BDOM = DOMS[:5]


def panel_d(C, labels=False):
    off = 16.5 if labels else 0.0
    if labels:
        row_labels(C, 15.8)
    late_band(C, 0.5, off + 88.0)
    for j, dm in enumerate(BDOM):
        xx = off + 3.0 + [0, 16.5, 28.5, 41.0, 58.0][j]
        C.rect(xx, 2.2, 2.0, 1.6, fc=DCOL[dm], ec='none')
        C.text(xx + 2.6, 3.0, dm, fontsize=5.2, va='center')
    C.line([off + 72.0, off + 72.0], [2.0, 4.0], color=INK, lw=0.9)
    C.text(off + 73.1, 3.0, 'Sum = M2 − M3', fontsize=5.2, va='center')
    C.text(off + 3.0, 6.6, 'Right of 0, the trait accounts for part of a higher age gap in the stage; '
                           'left of 0, for part of a lower age gap', fontsize=4.9, va='center', color=MID)
    kmm = 41.0
    spec = [(1, -0.30, 0.36, [-0.2, 0, 0.2]), (3, -0.36, 0.86, [-0.3, 0, 0.3, 0.6])]
    x = off + 3.0
    for s, lo, hi, ticks in spec:
        w = (hi - lo) * kmm
        ax = grid_axes(C, x, w)
        clean(ax, left=False, grid='x')
        ax.axvline(0, color=MID, lw=0.5, zorder=2)
        d = GB[GB.stage == s]
        for i, cl in enumerate(ROWS):
            e = d[d.clock == cl].set_index('domain')
            pos = neg = 0.0
            for dm in BDOM:
                v = e.loc[dm, 'contribution']
                if v >= 0:
                    ax.barh(ry(i), v, left=pos, height=RH * 0.64, color=DCOL[dm], lw=0, zorder=3)
                    pos += v
                else:
                    ax.barh(ry(i), v, left=neg, height=RH * 0.64, color=DCOL[dm], lw=0, zorder=3)
                    neg += v
            tot = e.explained.iloc[0]
            ax.plot([tot, tot], [ry(i) - RH * 0.42, ry(i) + RH * 0.42], color=INK, lw=0.9, zorder=5,
                    solid_capstyle='butt')
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.set_xticklabels([um(f'{t:g}') for t in ticks])
        C.text(x + w / 2, Y0 - 1.0, f'Stage {s} vs 0', fontsize=5.8, ha='center', va='bottom', color=STAGE_COL[s],
               fontweight='bold')
        x += w + 4.0
    C.text(off + 3.0 + (x - off - 7.0) / 2, YB + 4.6, 'Contribution to the change from M2 to M3 (s.d.)',
           fontsize=5.8, ha='center', va='top')


# ---------------------------------------------------------------- e  overlap of age gaps across stages
ECL = ['Kidney', 'Heart']


def panel_e(C):
    ay, ah, aw, gap, ax0 = 13.0, 34.0, 16.0, 1.4, 8.6
    for k, cl in enumerate(ECL):
        x = ax0 + k * (aw + gap)
        ax = C.ax(x, ay, aw, ah)
        clean(ax, left=False)
        d = DEN[DEN.clock == cl]
        mx = d.density.max()
        for s in range(4):
            e = d[d.stage == s]
            base = 3 - s
            yy = base + 1.55 * e.density.values / mx
            ax.fill_between(e.x, base, yy, color=lighten(STAGE_COL[s], 0.35), lw=0, zorder=10 - s)
            ax.plot(e.x, yy, color=STAGE_COL[s], lw=0.6, zorder=10 - s)
            ax.plot([e.x.min(), e.x.max()], [base, base], color=STAGE_COL[s], lw=0.4, zorder=10 - s)
        m3 = d[d.stage == 3]['median'].iloc[0]
        ax.axvline(m3, color=INK, lw=0.55, ls=(0, (2.2, 1.3)), zorder=12)
        ax.set_xlim(-3.0, 3.2)
        ax.set_ylim(-0.05, 4.75)
        ax.patch.set_alpha(0)
        ax.set_yticks([])
        ax.set_xticks([-2, 0, 2])
        ax.set_xticklabels([um('-2'), '0', '2'])
        C.text(x + 0.3, ay - 6.4, cl, fontsize=5.9, fontweight='bold', color=ORGAN_COL[cl], va='bottom')
        C.text(x + 0.3, ay - 0.6, um(f'{WS.loc[cl, "above_st3_median_stage12_pct"]:.1f}% of stages 1–2\n'
                                     f'above stage 3 median'), fontsize=4.7, color=MID, va='bottom',
               linespacing=1.0)
        if k == 0:
            for s in range(4):
                C.text(x - 0.6, ay + ah * (1 - (3 - s + 0.25) / 4.8), STAGE_LAB[s], fontsize=5.1, ha='right',
                       va='center', color=STAGE_COL[s] if s else MID)
    C.text(ax0 + aw + gap / 2, ay + ah + 4.6, 'Age gap adjusted for\nM2 covariates (s.d.)', fontsize=5.5,
           ha='center', va='top', linespacing=1.0)
    # all clocks: variance explained by stage
    rh, g2, y0 = 2.7, 1.2, ay

    def yr(i):
        return y0 + i * rh + (g2 if i >= NE else 0) + rh / 2
    yb = y0 + len(ROWS) * rh + g2
    lx = 54.6
    C.rect(lx - 10.8, yr(NE) - rh / 2 - 0.4, 66.0 - lx + 10.8 - 0.4, len(_l) * rh + 0.8, fc=LATE_BAND, ec='none',
           lw=0, zorder=0)
    for i, cl in enumerate(ROWS):
        C.text(lx, yr(i), cl, fontsize=4.8, ha='right', va='center', color=ORGAN_COL[cl])
    a1 = C.ax(lx + 0.8, y0, 10.0, yb - y0)
    clean(a1, left=False, grid='x')
    a1.patch.set_alpha(0)
    for i, cl in enumerate(ROWS):
        r = WS.loc[cl]
        a1.barh(yr(i), r.r2_stage_pct, height=rh * 0.62, color=lighten(ORGAN_COL[cl], 0.2), lw=0, zorder=3)
        a1.errorbar(r.r2_stage_pct, yr(i), xerr=[[r.r2_stage_pct - r.r2_lo], [r.r2_hi - r.r2_stage_pct]],
                    fmt='none', ecolor=INK, elinewidth=0.45, zorder=4)
    a1.set_ylim(yb, y0)
    a1.set_yticks([])
    a1.set_xlim(0, 4.0)
    a1.set_xticks([0, 2, 4])
    C.text(lx + 0.5, ay - 4.4, 'Variance explained\nby stage (%)', fontsize=5.1, ha='center', va='bottom',
           linespacing=1.0)


# ---------------------------------------------------------------- f  multi-organ ageing burden
BCOL = ['#E6E6E6', '#F2C9A8', '#D9732B', '#A4161A']


def panel_f(C):
    y0, rh = 13.0, 8.2
    ax = C.ax(9.0, y0, 21.0, 4 * rh)
    clean(ax, left=False)
    keys = ['organs_0', 'organs_1', 'organs_2', 'organs_3plus']
    for i, r in BD.iterrows():
        left = 0
        for kk, c in zip(keys, BCOL):
            v = 100 * r[kk]
            ax.barh(i, v, left=left, height=0.64, color=c, lw=0)
            if v > 12:
                ax.text(left + v / 2, i, f'{v:.0f}', fontsize=4.8, ha='center', va='center',
                        color='white' if kk in ('organs_2', 'organs_3plus') else INK)
            left += v
    ax.set_ylim(3.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel('Participants (%)', labelpad=1.2, fontsize=5.6)
    for i in range(4):
        C.text(8.4, y0 + (i + 0.5) * rh, STAGE_LAB[i], fontsize=5.2, ha='right', va='center',
               color=STAGE_COL[i] if i else MID)
        r = BD.iloc[i]
        C.text(36.2, y0 + (i + 0.5) * rh, f'{100 * (r.organs_2 + r.organs_3plus):.1f}', fontsize=5.3,
               ha='right', va='center', color=ACCENT, fontweight='bold')
        C.text(57.6, y0 + (i + 0.5) * rh, f'{BR.iloc[i].ge2_pct:.1f}', fontsize=5.3, ha='right', va='center',
               color=BLUE, fontweight='bold')
    C.text(36.2, y0 - 0.8, '≥2 (%)', fontsize=5.0, ha='right', va='bottom', color=ACCENT)
    C.text(57.6, y0 - 0.8, '≥2 (%),\nkidney-\nresidualised', fontsize=4.8, ha='right', va='bottom', color=BLUE,
           linespacing=0.95)
    a2 = C.ax(38.0, y0, 12.5, 4 * rh)
    clean(a2, left=False, grid='x')
    a2.axvline(1, color=LIGHT, lw=0.5)
    for i in [1, 2, 3]:
        for m, dy, filled in [('M2', -0.15, True), ('M3', 0.17, False)]:
            r = OR[(OR.model == m) & (OR.stage == i)].iloc[0]
            ci(a2, np.array([r.OR]), np.array([i + dy]), np.array([r.lo]), np.array([r.hi]), color=STAGE_COL[i],
               filled=filled, ms=2.2, lw=0.6)
    a2.text(1.0, 0, 'Ref.', fontsize=4.8, color=MID, ha='center', va='center', zorder=5,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
    a2.set_xscale('log')
    a2.set_xlim(0.7, 6.0)
    a2.set_xticks([1, 2, 4])
    a2.set_xticklabels(['1', '2', '4'])
    a2.minorticks_off()
    a2.set_ylim(3.5, -0.5)
    a2.set_yticks([])
    a2.set_xlabel('OR, ≥2 vs stage 0', labelpad=1.2, fontsize=5.6)
    C.line([38.6], [y0 - 5.4], marker='o', ms=2.2, color=INK, lw=0)
    C.text(39.6, y0 - 5.4, 'M2', fontsize=5.0, va='center')
    C.line([38.6], [y0 - 2.4], marker='o', ms=2.2, mfc='white', mec=INK, mew=0.6, lw=0)
    C.text(39.6, y0 - 2.4, 'M3', fontsize=5.0, va='center')
    ly = y0 + 4 * rh + 9.4
    C.text(9.0, ly, 'Aged target organs', fontsize=5.0, color=MID, va='center')
    for kk, (c, l) in enumerate(zip(BCOL, ['0', '1', '2', '≥3'])):
        C.rect(27.5 + kk * 5.6, ly - 0.8, 1.7, 1.6, fc=c, ec='none')
        C.text(29.7 + kk * 5.6, ly, l, fontsize=5.0, va='center')
    n12 = int(BD[BD.stage.isin([1, 2])].n_ge2.sum())
    n3 = int(BD[BD.stage == 3].n.iloc[0])
    C.text(9.0, ly + 3.6, f'≥2 aged organs in stages 1–2: n = {n12:,}\n(all of stage 3: n = {n3:,})', fontsize=5.0,
           va='top', color=INK, linespacing=1.05)


# ---------------------------------------------------------------- g  identification of high burden
GM = [('Age, sex', 'Age, sex', LIGHT), ('+ CKM stage', '+ CKM stage', STAGE_COL[1]),
      ('+ PREVENT risk', '+ PREVENT', STAGE_COL[2]),
      ('+ clinical panel without eGFR', '+ panel without eGFR', '#86A6CF'),
      ('+ eGFR (full clinical panel)', '+ eGFR (full panel)', BLUE)]


def panel_g(C):
    n12 = int(BD[BD.stage.isin([1, 2])].n.sum())
    nh = int(BD[BD.stage.isin([1, 2])].n_ge2.sum())
    C.text(6.0, 2.8, f'Stages 1–2: n = {n12:,}; high burden (≥2), n = {nh:,}', fontsize=5.0, va='center',
           color=MID)
    ax = C.ax(7.5, 7.0, 20.0, 20.0)
    clean(ax)
    ax.plot([0, 1], [0, 1], color=FAINT, lw=0.5, ls='--')
    for k, (m, lab, c) in enumerate(GM):
        d = ROC[ROC.model == m]
        ax.plot(d.fpr, d.tpr, color=c, lw=0.9 if k < 4 else 1.1, zorder=3 + k)
        yy = 8.2 + k * 3.3
        C.line([30.0, 32.2], [yy, yy], color=c, lw=1.1)
        C.text(32.8, yy, lab, fontsize=4.9, va='center')
        C.text(55.6, yy, f'{AUC.loc[m, "auc"]:.3f}', fontsize=5.0, va='center', ha='right',
               color=c if k else MID, fontweight='bold')
    C.text(55.6, 5.0, 'AUC', fontsize=5.0, ha='right', va='center', color=MID)
    r = AUC.loc['+ clinical panel without eGFR']
    C.text(30.0, 25.0, um(f'Adding eGFR:\nΔAUC {r.delta_full_minus_this:.3f} ({r.d_lo:.3f}–{r.d_hi:.3f})'),
           fontsize=4.9, va='top', color=BLUE, linespacing=1.05)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_xlabel('1 − specificity', labelpad=1.0, fontsize=5.6)
    ax.set_ylabel('Sensitivity', labelpad=1.0, fontsize=5.6)
    # standardised mean differences
    d = SMD[SMD.trait != 'Age'].sort_values('smd').reset_index(drop=True)
    lab = {'PREVENT 10-y CVD risk': 'PREVENT risk', 'Waist': 'Waist circumference', 'SBP': 'Systolic BP'}
    y1, rr = 37.0, 1.95
    a2 = C.ax(17.5, y1, 21.0, len(d) * rr)
    clean(a2, left=False, grid='x')
    a2.axvline(0, color=LIGHT, lw=0.5)
    for i, r in d.iterrows():
        c = BLUE if r.smd < 0 else ACCENT
        a2.plot([0, r.smd], [i, i], color=lighten(c, 0.4), lw=0.8)
        a2.scatter(r.smd, i, s=6, color=c, zorder=3, lw=0)
        C.text(16.8, y1 + (i + 0.5) * rr, lab.get(r.trait, r.trait), fontsize=4.8, ha='right', va='center')
    a2.set_ylim(len(d) - 0.5, -0.5)
    a2.set_yticks([])
    a2.set_xlim(-0.7, 0.3)
    a2.set_xticks([-0.6, -0.3, 0, 0.3])
    a2.set_xticklabels([um('-0.6'), um('-0.3'), '0', '0.3'])
    a2.set_xlabel('SMD, high vs low burden', labelpad=1.2, fontsize=5.6)
    hp = PC[(PC.high.str.startswith('High')) & (PC.prev_cat == '<5%')].pct.iloc[0]
    lp = PC[(PC.high.str.startswith('Low')) & (PC.prev_cat == '<5%')].pct.iloc[0]
    C.text(40.5, y1 + 0.6, f'PREVENT <5%:\n{hp:.1f}% of high\nvs {lp:.1f}% of\nlow burden', fontsize=4.8,
           color=MID, va='top', ha='left', linespacing=1.02)


# ---------------------------------------------------------------- layout and export
W, H = 180, 200
PANELS = {
    'a': (panel_a, 0, 0, 120, 74),
    'b': (panel_b, 121, 0, 59, 74),
    'c': (panel_c, 0, 75, 91, 61),
    'd': (panel_d, 91, 75, 89, 61),
    'e': (panel_e, 0, 137, 66, 63),
    'f': (panel_f, 66, 137, 58, 63),
    'g': (panel_g, 124, 137, 56, 63),
}
LABELLED = {'d': 16.5}      # the stand-alone export of d carries its own clock labels


def export():
    pd_ = os.path.join(ROOT, 'panels')
    os.makedirs(pd_, exist_ok=True)
    for f in os.listdir(pd_):
        os.remove(os.path.join(pd_, f))
    for lab, (fn, x, y, w, h) in PANELS.items():
        extra = LABELLED.get(lab, 0.0)
        fig = plt.figure(figsize=((w + extra) * MM, h * MM))
        C = Canvas(fig, 0, 0, w + extra, h)
        fn(C, labels=True) if extra else fn(C)
        letter(C, lab)
        _save(fig, os.path.join(pd_, f'Fig2{lab}'))
        plt.close(fig)
    fig = plt.figure(figsize=(W * MM, H * MM))
    for lab, (fn, x, y, w, h) in PANELS.items():
        C = Canvas(fig, x, y, W, H)
        fn(C)
        letter(C, lab)
    _save(fig, os.path.join(ROOT, 'Fig2'), tif=True)
    plt.close(fig)


if __name__ == '__main__':
    export()
    print('Figure 2 written')
