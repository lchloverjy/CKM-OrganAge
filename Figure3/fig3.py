"""Figure 3. Organ age gaps are associated with incident disease of the corresponding organ.

Reads only ../source_data/*.csv.  Run:  python fig3.py
"""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style4 import *
import pandas as pd, numpy as np
from matplotlib.colors import TwoSlopeNorm

SD = os.path.join(HERE, '..', 'source_data') + '/'
ROOT = os.path.join(HERE, '..')
At = pd.read_csv(SD + 'Fig3a_atlas_tier2_model_b.csv')
Bf = pd.read_csv(SD + 'Fig3b_tier1_paired_hr.csv')
Br = pd.read_csv(SD + 'Fig3b_tier1_rank_probability.csv')
Ci = pd.read_csv(SD + 'Fig3c_clock_correspondence_index.csv')
Cd = pd.read_csv(SD + 'Fig3c_disease_correspondence.csv')
Dc = pd.read_csv(SD + 'Fig3d_phenome_by_clock.csv')
Ef = pd.read_csv(SD + 'Fig3e_paired_raw_vs_residualised.csv')

PAIRS = [('HF', 'Heart'), ('CKD', 'Kidney'), ('DEM', 'Brain'), ('ASCVD', 'Artery'), ('T2D', 'Pancreas'), ('CLD', 'Liver')]
OLAB = {'HF': 'Heart failure', 'CKD': 'CKD', 'DEM': 'Dementia', 'ASCVD': 'ASCVD', 'T2D': 'Type 2 diabetes',
        'CLD': 'Chronic liver disease'}
MSHADE = {'a': '#C2C8CF', 'b': '#8C96A1', 'c': '#56616D', 'd': '#151B22'}
ROWS = ['Heart', 'Artery', 'Kidney', 'Pancreas', 'Liver', 'Brain', 'Lung', 'Immune', 'Muscle', 'Intestine',
        'Organismal', 'Conventional']


def logx(ax, lo, hi, ticks):
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def _index_order():
    ix = Ci[Ci.version == 'raw'].set_index('clock')['index'].sort_values(ascending=False)
    return ix.index.tolist()


# ---------------------------------------------------------------------------------------------- a  atlas
def panel_a(C):
    X0, CW, GAP, Y0, CH = 17.5, 3.2, 0.9, 13.0, 2.7
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.8)
    cols = At.drop_duplicates('outcome').sort_values('column_order')
    xs, groups, x, prev = {}, [], X0, None
    for r in cols.itertuples():
        if prev is not None and r.column_group != prev:
            x += GAP
        if r.column_group != prev:
            groups.append([r.column_group, x, x])
        xs[r.outcome] = x
        groups[-1][2] = x + CW
        x += CW
        prev = r.column_group
    def ry(i):
        return Y0 + i * CH + (0.7 if i >= 10 else 0)
    # cells
    for r in At.itertuples():
        i = ROWS.index(r.clock)
        v = norm(r.log_hr)
        C.rect(xs[r.outcome], ry(i), CW, CH, fc=DIV(v), ec='white', lw=0.3)
        if r.q_bh_552 < 0.05:
            C.line([xs[r.outcome] + CW / 2], [ry(i) + CH / 2], marker='o', ms=1.05, lw=0,
                   color='white' if (v > 0.82 or v < 0.12) else INK)
    for r in At[At.paired].itertuples():
        C.rect(xs[r.outcome], ry(ROWS.index(r.clock)), CW, CH, fc='none', ec=INK, lw=0.85)
    for i, c in enumerate(ROWS):
        C.text(X0 - 0.8, ry(i) + CH / 2, c, ha='right', va='center', fontsize=5.4, color=ORGAN_COL[c])
    # group headers
    for g, a, b in groups:
        col = ORGAN_COL.get(g, MID)
        C.line([a + 0.15, b - 0.15], [Y0 - 1.1] * 2, color=col, lw=0.9, solid_capstyle='butt')
        if g != 'Control':
            C.text((a + b) / 2, Y0 - 1.7, g, ha='center', va='bottom', fontsize=5.0, color=col)
    # rank row
    yr = ry(11) + CH + 1.9
    C.text(X0 - 0.8, yr, 'Rank of paired clock', ha='right', va='center', fontsize=5.0, color=MID)
    for r in cols.itertuples():
        if np.isnan(r.paired_rank_among_10):
            s, kw = '–', dict(color=LIGHT)
        else:
            k = int(r.paired_rank_among_10)
            s, kw = str(k), (dict(color=INK, fontweight='bold') if k == 1 else dict(color=MID))
        C.text(xs[r.outcome] + CW / 2, yr, s, ha='center', va='center', fontsize=5.0, **kw)
    # disease labels
    for r in cols.itertuples():
        kw = dict(fontweight='bold') if r.tier1 else {}
        if r.outcome == 'NEG':
            kw = dict(color=MID, fontstyle='italic')
        C.text(xs[r.outcome] + CW / 2 + 0.6, yr + 1.9, r.label, ha='right', va='top', fontsize=5.0, rotation=52,
               rotation_mode='anchor', **kw)
    # legend and colour bar
    C.rect(X0 + 0.2, 2.2, 2.4, 2.0, fc='white', ec=INK, lw=0.85)
    C.text(X0 + 3.4, 3.2, 'Paired (clinically prespecified) clock', fontsize=5.2, va='center')
    C.line([X0 + 44.4], [3.2], marker='o', ms=1.6, lw=0, color=INK)
    C.text(X0 + 45.8, 3.2, 'FDR < 0.05 (552 tests; control, 12 tests)', fontsize=5.2, va='center')
    C.text(X0 + 93.5, 3.2, 'Bold, tier-1 disease', fontsize=5.2, va='center', fontweight='bold')
    hcolorbar(C, DIV, norm, 146, 4.2, 30, 1.3, 'log HR per s.d. (model b)', ticks=[-0.3, 0, 0.4, 0.8],
              fmt=lambda t: um(f'{t:g}'))


# ---------------------------------------------------------------------------------------------- b  tier 1
def panel_b(C):
    ax = C.ax(27, 10, 40, 35)
    clean(ax, left=False, grid='x')
    ax.axvline(1, color=LIGHT, lw=0.45)
    n = len(PAIRS)
    for i, (k, o) in enumerate(PAIRS):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#F7F7F7', lw=0, zorder=0)
        for j, m in enumerate('abcd'):
            r = Bf[(Bf.outcome == k) & (Bf.clock == o) & (Bf.model == m)].iloc[0]
            y = i - 0.3 + j * 0.2
            sig = r.p_holm < 0.05
            ax.plot([r.lo, r.hi], [y, y], color=MSHADE[m], lw=0.8, zorder=2)
            ax.scatter(r.hr, y, s=7, fc=MSHADE[m] if sig else 'white', ec=MSHADE[m], lw=0.6, zorder=3)
        yc = 10 + 35 * (i + 0.5) / n
        C.text(26, yc - 1.3, OLAB[k], ha='right', va='center', fontsize=5.6, fontweight='bold')
        C.text(26, yc + 1.3, f'{o} clock', ha='right', va='center', fontsize=5.3, color=ORGAN_COL[o])
        pr = Br[(Br.outcome == k)].p_rank1.iloc[0]
        C.text(76.5, yc, f'{pr:.2f}', ha='center', va='center', fontsize=5.3,
               color=INK if pr >= 0.5 else MID, fontweight='bold' if pr >= 0.5 else 'normal')
    C.text(76.5, 9.2, 'P(rank 1)', ha='center', va='bottom', fontsize=5.0, color=MID)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    logx(ax, 0.85, 2.5, [1, 1.5, 2, 2.5])
    ax.set_xlabel('HR per s.d. (95% CI)', labelpad=1.5)
    MLAB = {'a': 'a  Base covariates', 'b': 'b  + CKM stage', 'c': 'c  + stage-defining traits',
            'd': 'd  + eGFRcr-cys, UACR, NT-proBNP, TNNI3'}
    for j, m in enumerate('abcd'):
        xx, yy = (5.0 + (j % 2) * 30.0, 2.0 + (j // 2) * 3.0)
        C.line([xx + 0.8], [yy], marker='o', ms=2.6, color=MSHADE[m], lw=0)
        C.text(xx + 2.2, yy, MLAB[m], fontsize=5.0, va='center')
    C.text(84, 51.8, 'Open symbols,\nHolm-adjusted P ≥ 0.05', fontsize=5.0, color=MID, ha='right', va='bottom',
           linespacing=1.0)


# ---------------------------------------------------------------------------------------------- c  correspondence index
def panel_c(C):
    order = _index_order()
    n = len(order)
    AX, AY, AW, AH = 24, 10, 51, 34
    ax = C.ax(AX, AY, AW, AH)
    clean(ax, left=False, grid='x')
    ax.axvline(1, color=LIGHT, lw=0.5, zorder=1)
    raw = Ci[Ci.version == 'raw'].set_index('clock')
    res = Ci[Ci.version == 'resid'].set_index('clock')
    for i, c in enumerate(order):
        col = ORGAN_COL[c]
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#F7F7F7', lw=0, zorder=0)
        d = Cd[Cd.paired_clock == c].sort_values('excess_log_hr_raw').reset_index(drop=True)
        m = len(d)
        for k, r in d.iterrows():
            yy = i - 0.27 + (k - (m - 1) / 2) * (0.16 / max(m - 1, 1))
            top = r.rank_among_10_raw == 1
            ax.scatter(np.exp(r.excess_log_hr_raw), yy, s=5.5 if top else 4, zorder=2,
                       fc=col if top else lighten(col, 0.55), ec='white' if top else 'none', lw=0.25)
        r0 = raw.loc[c]
        ax.plot([np.exp(r0.lo), np.exp(r0.hi)], [i + 0.02] * 2, color=col, lw=1.1, zorder=3, solid_capstyle='butt')
        ax.scatter(np.exp(r0['index']), i + 0.02, s=15, fc=col, ec=INK, lw=0.45, zorder=4)
        r1 = res.loc[c]
        ax.plot([np.exp(r1.lo), np.exp(r1.hi)], [i + 0.31] * 2, color=lighten(col, 0.35), lw=0.55, zorder=3)
        ax.scatter(np.exp(r1['index']), i + 0.31, s=9, marker='D', fc='white', ec=col, lw=0.6, zorder=4)
        yc = AY + AH * (i + 0.5) / n
        sig = r0.q_bh < 0.05
        C.text(AX - 1, yc, c, ha='right', va='center', fontsize=5.5, color=col, fontweight='bold' if sig else 'normal')
        C.text(81.5, yc, f'{int(r0.n_rank1)}/{int(r0.n_diseases)}', ha='center', va='center', fontsize=5.4,
               color=INK if sig else MID, fontweight='bold' if sig else 'normal')
        C.text(89.0, yc, fmt_p(r0.q_bh) if r0.q_bh < 0.001 else f'{r0.q_bh:.2f}' if r0.q_bh >= 0.01 else f'{r0.q_bh:.3f}',
               ha='center', va='center', fontsize=5.0, color=INK if sig else MID)
    C.text(81.5, 9.1, 'Ranked\nfirst', ha='center', va='bottom', fontsize=5.0, color=MID, linespacing=0.95)
    C.text(89.0, 9.1, 'FDR P', ha='center', va='bottom', fontsize=5.0, color=MID)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    logx(ax, 0.85, 2.3, [0.9, 1, 1.25, 1.5, 2])
    ax.set_xlabel('HR ratio, paired clock vs other nine organ clocks', labelpad=1.5)
    C.line([5.0], [2.3], marker='o', ms=1.9, color=lighten(INK, 0.5), lw=0)
    C.text(6.5, 2.3, 'Single disease (solid, paired clock ranked first of ten)', fontsize=5.0, va='center')
    C.line([5.0], [5.6], marker='o', ms=3.0, mfc=MID, mec=INK, mew=0.45, lw=0)
    C.text(6.5, 5.6, 'Clock index (95% CI)', fontsize=5.0, va='center')
    C.line([33.0], [5.6], marker='D', ms=2.3, mfc='white', mec=MID, mew=0.6, lw=0)
    C.text(34.5, 5.6, 'After kidney-function residualisation', fontsize=5.0, va='center')


# ---------------------------------------------------------------------------------------------- d  phenome by clock
def panel_d(C):
    order = _index_order() + ['Organismal', 'Conventional']
    D = Dc.set_index('clock')
    n = len(order)
    AX, AY, AW, AH = 15, 9.5, 25, 34.5
    nmax = float(D.loc[order, 'n_sig'].max())
    step = 20 if nmax <= 50 else (50 if nmax <= 160 else 100)
    XMAX = max(50.0, np.ceil(nmax * 1.24 / 10.0) * 10.0)
    ax = C.ax(AX, AY, AW, AH)
    clean(ax, left=False, grid='x')
    for i, c in enumerate(order):
        r = D.loc[c]
        col = ORGAN_COL[c]
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#F7F7F7', lw=0, zorder=0)
        own = 0 if np.isnan(r.n_sig_corresponding) else int(r.n_sig_corresponding)
        ax.barh(i, own, height=0.62, color=col, lw=0, zorder=2)
        ax.barh(i, r.n_sig - own, left=own, height=0.62, color=lighten(col, 0.62), lw=0, zorder=2)
        lab = f'{own}/{int(r.n_sig)}' if not np.isnan(r.n_sig_corresponding) else f'{int(r.n_sig)}'
        ax.text(r.n_sig + XMAX * 0.016, i, lab, va='center', ha='left', fontsize=5.0, color=MID)
        yc = AY + AH * (i + 0.5) / n
        C.text(AX - 1, yc, c, ha='right', va='center', fontsize=5.5, color=col)
        hit = r.top_in_corresponding_chapter == True
        C.text(48.5, yc, r.top_label, ha='left', va='center', fontsize=5.2,
               color=col if hit else MID, fontweight='bold' if hit else 'normal')
        C.text(87.5, yc, f'{r.top_hr:.2f}', ha='right', va='center', fontsize=5.2, color=INK if hit else MID)
    C.text(48.5, 8.6, 'Strongest phecode', ha='left', va='bottom', fontsize=5.0, color=MID)
    C.text(87.5, 8.6, 'HR', ha='right', va='bottom', fontsize=5.0, color=MID)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(0, XMAX)
    ax.set_xticks(np.arange(0, XMAX * 0.86, step))
    ax.set_xlabel(f'FDR-significant phecodes\n(of {int(Dc.n_phecodes.max())}; model c)', labelpad=1.5, linespacing=1.0)
    C.rect(5.0, 1.4, 2.6, 1.8, fc=MID, ec='none', lw=0)
    C.text(8.4, 2.3, 'Corresponding chapter', fontsize=5.0, va='center')
    C.rect(33.0, 1.4, 2.6, 1.8, fc=lighten(MID, 0.62), ec='none', lw=0)
    C.text(36.4, 2.3, 'Other chapters', fontsize=5.0, va='center')
    C.text(5.0, 5.6, 'Coloured bold, strongest phecode in the corresponding chapter', fontsize=5.0, va='center',
           color=MID)


# ---------------------------------------------------------------------------------------------- e  kidney residualisation
LBL = {'HF': ('Heart failure', 1.33, 1.64, 'right'), 'AF': ('AF', 1.47, 1.79, 'right'),
       'CKD': ('CKD', 2.06, 1.36, 'left'), 'AKI': ('AKI', 1.70, 1.21, 'left'),
       'DEM': ('Dementia', 1.62, 2.02, 'right'), 'AD': ('AD', 2.08, 1.86, 'left'),
       'VAD': ('Vascular dementia', 1.92, 2.34, 'right'), 'ILD': ('ILD', 1.98, 1.63, 'left'),
       'GOUT': ('Gout', 1.70, 1.12, 'left')}


def panel_e(C):
    ax = C.ax(13, 9, 38, 36)
    clean(ax)
    lo_, hi_ = 0.9, 2.45
    ax.plot([lo_, hi_], [lo_, hi_], color=MID, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.axhline(1, color=FAINT, lw=0.4, zorder=0); ax.axvline(1, color=FAINT, lw=0.4, zorder=0)
    ns = Ef[Ef.q_bh_552_raw >= 0.05]
    ax.scatter(ns.hr_raw, ns.hr_resid, s=5, fc='white', ec=LIGHT, lw=0.45, zorder=2)
    sg = Ef[Ef.q_bh_552_raw < 0.05]
    ax.scatter(sg.hr_raw, sg.hr_resid, s=11, fc=[ORGAN_COL[c] for c in sg.paired_clock], ec='white', lw=0.3, zorder=3)
    t1 = Ef[Ef.tier1 & (Ef.q_bh_552_raw < 0.05)]
    ax.scatter(t1.hr_raw, t1.hr_resid, s=22, fc='none', ec=INK, lw=0.55, zorder=4)
    for k, (txt, tx, ty, ha) in LBL.items():
        r = Ef[Ef.outcome == k].iloc[0]
        ax.annotate(txt, (r.hr_raw, r.hr_resid), xytext=(tx, ty), fontsize=5.0, ha=ha, va='center', color=INK,
                    arrowprops=dict(arrowstyle='-', lw=0.3, color=MID, shrinkA=0.3, shrinkB=1.6))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    tk = [1, 1.5, 2]
    ax.set_xticks(tk); ax.set_xticklabels([f'{t:g}' for t in tk])
    ax.set_yticks(tk); ax.set_yticklabels([f'{t:g}' for t in tk])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator()); ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel('HR per s.d., original clock')
    ax.set_ylabel('HR per s.d., kidney-function-\nresidualised clock', linespacing=1.0)
    S = Ef[Ef.q_bh_552_raw < 0.05].copy()
    med = S.groupby('paired_clock').attenuation_pct.median().sort_values()
    bx = C.ax(66, 11, 20, 32)
    clean(bx, left=False, grid='x')
    bx.axvline(0, color=LIGHT, lw=0.5)
    for i, c in enumerate(med.index):
        d = S[S.paired_clock == c]
        bx.scatter(d.attenuation_pct, [i] * len(d), s=6, fc=lighten(ORGAN_COL[c], 0.3), ec='white', lw=0.25, zorder=2)
        bx.plot([med[c]] * 2, [i - 0.32, i + 0.32], color=ORGAN_COL[c], lw=1.1, zorder=3)
        C.text(65, 11 + 32 * (i + 0.5) / len(med), f'{c} ({len(d)})', ha='right', va='center', fontsize=5.0,
               color=ORGAN_COL[c])
    bx.set_ylim(len(med) - 0.5, -0.5); bx.set_yticks([])
    bx.set_xlim(-15, 65); bx.set_xticks([0, 25, 50])
    bx.set_xlabel('Attenuation of\nlog HR (%)', linespacing=1.0)
    C.text(66, 8.4, 'Bar, median; dot, disease', fontsize=4.9, color=MID, va='center')
    C.line([13.5], [2.4], marker='o', ms=2.6, color=MID, lw=0, mec='white', mew=0.3)
    C.text(15.0, 2.4, 'FDR < 0.05, original', fontsize=5.0, va='center')
    C.line([36.0], [2.4], marker='o', ms=2.1, mfc='white', mec=LIGHT, mew=0.45, lw=0)
    C.text(37.5, 2.4, 'FDR ≥ 0.05', fontsize=5.0, va='center')
    C.line([50.5], [2.4], marker='o', ms=3.4, mfc='none', mec=INK, mew=0.55, lw=0)
    C.text(52.0, 2.4, 'Tier 1', fontsize=5.0, va='center')
    C.line([60.0, 63.5], [2.4, 2.4], color=MID, lw=0.5, ls=(0, (3, 2)))
    C.text(64.3, 2.4, 'No attenuation', fontsize=5.0, va='center')


W, H = 180, 183
PANELS = {
    'a': (panel_a, 0, 0, 180, 72),
    'b': (panel_b, 0, 74, 85, 53),
    'c': (panel_c, 86, 74, 94, 53),
    'd': (panel_d, 0, 129, 89, 54),
    'e': (panel_e, 90, 129, 90, 54),
}

if __name__ == '__main__':
    for f in glob.glob(os.path.join(ROOT, 'panels', 'Fig3*')):
        os.remove(f)
    build(PANELS, W, H, 'Fig3', ROOT)
    print('Fig3 done')
