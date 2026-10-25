"""Figure 1 (v2): graphical-abstract-style study overview.

Reads only ../source_data/. Each panel is drawn by one function that is used both for the single-panel export
(../panels/) and for the composite (../Fig1.pdf/.png/.tif).
Usage: python fig1v2.py            # all panels + composite
       python fig1v2.py a b        # selected single panels only
"""
import os, sys, textwrap
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from style4 import *
import ga
import matplotlib.patheffects as pe_
from ga import icon, rrect, circle, chevron, kicker, title

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, '..', 'source_data')
ROOT = os.path.join(HERE, '..')


def rd(f, **kw):
    return pd.read_csv(os.path.join(SD, f), **kw)


SLATE = '#3B4A5A'
ICON = {'Heart': 'heart-organ', 'Artery': 'blood-vessel', 'Kidney': 'kidneys', 'Liver': 'liver',
        'Pancreas': 'pancreas', 'Brain': 'neurology', 'Lung': 'lungs', 'Immune': 'antibody',
        'Intestine': 'intestine', 'Muscle': 'weights', 'Organismal': 'body', 'Conventional': 'blood-drop'}


def mri(C, x, y, s, color=SLATE):
    """Simple MRI-scanner glyph in a box of size s (top-left x, y)."""
    cx, cy = x + s * 0.5, y + s * 0.45
    circle(C, cx, cy, s * 0.42, fc=color, ec='none', zorder=5)
    circle(C, cx, cy, s * 0.20, fc='white', ec='none', zorder=6)
    rrect(C, x + s * 0.02, cy - s * 0.035, s * 0.96, s * 0.11, r=s * 0.03, fc=color, ec='none', zorder=7)
    C.rect(x + s * 0.18, y + s * 0.86, s * 0.64, s * 0.08, fc=color, ec='none', zorder=5)


def wrap(s, n):
    return '\n'.join(textwrap.fill(p, n) for p in s.split('\n'))


# ======================================================================================== a
def pa(C):
    A = rd('Fig1a_cohort_data_layers.csv').set_index('key').value
    kicker(C, 4.2, 0.5, 'Cohort and data')
    title(C, 4.2, 3.3, f"{int(float(A.n_cohort)):,} adults, four data layers")
    # timeline
    x0, x1, y0, y1 = 3.0, 57.5, 2006, 2022
    X = lambda t: x0 + (t - y0) / (y1 - y0) * (x1 - x0)
    yt = 12.8
    C.line([X(2006), X(2021.75)], [yt, yt], color=LIGHT, lw=0.6)
    C.line([X(2006.3), X(2010.5)], [yt, yt], color=INK, lw=3.0, solid_capstyle='butt')
    C.line([X(2021.75)] * 2, [yt - 1.1, yt + 1.1], color=INK, lw=0.7)
    C.text(X(2008.4), yt - 1.9, 'Baseline', fontsize=5.2, ha='center', va='bottom', fontweight='bold')
    # imaging visit marker
    xm = X(2014.3)
    from matplotlib.path import Path
    ga_d = 0.9
    C.line([xm - ga_d, xm, xm + ga_d, xm, xm - ga_d], [yt, yt - ga_d, yt, yt + ga_d, yt], color=SLATE, lw=0.6)
    fu = float(A.fu_median_years)
    C.text(X(2016.4), yt - 1.9, f'Follow-up, median {fu:.1f} y', fontsize=5.2, ha='center', va='bottom')
    for t, lab, ha in [(2006, '2006', 'center'), (2010.5, '2010', 'center'), (2014.3, 'MRI, 2014\u2013', 'center'),
                       (2021.95, '2021', 'right')]:
        C.text(X(t), yt + 1.4, lab, fontsize=5.0, ha=ha, color=SLATE if 'MRI' in lab else MID)
    # data-layer cards
    layers = [
        ('test-tubes', 'Plasma proteome', '12 clocks', 'Olink Explore 3072 \u00b7 2,923 proteins'),
        ('medical-records', 'Clinical assessment', '4 stages', 'CKM stage, PREVENT, eGFR, NT-proBNP, TNNI3'),
        ('hospital', 'Hospital and death records', f"{int(float(A.deaths)):,} deaths",
         f"ICD-10 \u00b7 {float(A.person_years):,.0f} person-years"),
        ('MRI', 'Imaging visit', f"n = {int(float(A.n_imaging_any)):,}", 'Cardiac, aortic and brain MRI'),
    ]
    y = 18.4
    for ic, t1, key, det in layers:
        rrect(C, 2.5, y, 55.5, 7.4, r=1.2, fc='#F5F6F7', ec='none', zorder=1)
        if ic == 'MRI':
            mri(C, 4.2, y + 1.0, 5.4)
        else:
            icon(C, ic, 4.2, y + 1.0, 5.4, color=SLATE)
        C.text(11.6, y + 1.25, t1, fontsize=5.8, fontweight='bold')
        C.text(56.8, y + 1.25, key, fontsize=5.8, fontweight='bold', ha='right', color=BLUE)
        C.text(11.6, y + 4.05, det, fontsize=5.0, color=MID)
        y += 8.7
    # selection funnel
    yb = 54.6
    blocks = [(f"{int(float(A.n_ukb)):,}", 'UK Biobank'), (f"{int(float(A.n_gap)):,}", 'with proteomics'),
              (f"{int(float(A.n_cohort)):,}", 'CKM stages 0\u20133')]
    bw, gap = 16.6, 2.85
    for i, (num, lab) in enumerate(blocks):
        xb = 2.5 + i * (bw + gap)
        last = i == 2
        rrect(C, xb, yb, bw, 6.0, r=1.0, fc=SLATE if last else 'white', ec=SLATE, lw=0.5, zorder=2)
        C.text(xb + bw / 2, yb + 0.7, num, fontsize=6.0, fontweight='bold', ha='center',
               color='white' if last else INK, zorder=3)
        C.text(xb + bw / 2, yb + 3.4, lab, fontsize=5.0, ha='center', color='white' if last else MID, zorder=3)
        if i < 2:
            chevron(C, xb + bw + gap / 2 + 0.6, yb + 3.0, 2.2, w=1.0, color=LIGHT, lw=0.7)
    C.text(58.0, yb + 6.8, 'Supplementary Fig. 1, Table 1', fontsize=5.0, color=MID, ha='right', style='italic')


# ======================================================================================== b
def pb(C):
    B = rd('Fig1b_organ_clocks.csv').set_index('clock')
    kicker(C, 4.2, 0.5, 'Exposure')
    title(C, 4.2, 3.3, '12 proteomic organ clocks')
    rows = [['Heart', 'Artery', 'Kidney', 'Liver', 'Pancreas'], ['Brain', 'Lung', 'Immune', 'Intestine', 'Muscle']]
    cw = 12.0
    for r, row in enumerate(rows):
        y = 9.4 + r * 13.6
        for i, o in enumerate(row):
            cx = 1.0 + cw * i + cw / 2
            col = ORGAN_COL[o]
            circle(C, cx, y + 4.2, 4.1, fc=lighten(col, 0.86), ec=col if B.loc[o, 'target'] else 'none',
                   lw=0.75, zorder=2)
            icon(C, ICON[o], cx - 2.9, y + 1.3, 5.8, color=col)
            C.text(cx, y + 9.0, o, fontsize=5.4, fontweight='bold', ha='center', color=col)
            C.text(cx, y + 11.2, f"{B.loc[o, 'sd_years']:.1f} y", fontsize=5.0, ha='center', color=MID)
    # legend for target ring
    yl = 37.0
    circle(C, 3.0, yl + 1.0, 1.0, fc='white', ec=MID, lw=0.7)
    C.text(5.0, yl + 0.1, 'Target organ', fontsize=5.0, color=MID)
    C.text(61.0, yl + 0.1, 'Years per 1 s.d. of age gap', fontsize=5.0, color=MID, ha='right')
    # whole-body clocks
    yw = 40.3
    for i, (o, sub) in enumerate([('Organismal', 'non-enriched proteins'), ('Conventional', 'all proteins')]):
        xw = 1.0 + i * 30.2
        col = ORGAN_COL[o]
        rrect(C, xw, yw, 29.4, 7.0, r=1.2, fc='#F5F6F7', ec='none')
        icon(C, ICON[o], xw + 1.0, yw + 0.9, 5.2, color=col)
        C.text(xw + 7.2, yw + 0.9, o, fontsize=5.4, fontweight='bold', color=col if o == 'Organismal' else MID)
        C.text(xw + 28.6, yw + 0.95, f"{B.loc[o, 'sd_years']:.1f} y", fontsize=5.0, color=MID, ha='right')
        C.text(xw + 7.2, yw + 3.6, sub, fontsize=5.0, color=MID)
    # pipeline
    yp = 49.6
    steps = [('1  Select', 'organ-enriched\nproteins'),
             ('2  Train', 'LightGBM,\nSHAP-Boruta'),
             ('3  Age gap', 'age residual,\nz-scored'),
             ('4  Aged organ', '>90th centile\nof stage 0')]
    bw, gap = 14.1, 1.2
    yp = 51.4
    for i, (h, t) in enumerate(steps):
        xb = 1.0 + i * (bw + gap)
        rrect(C, xb, yp, bw, 9.0, r=1.0, fc='white', ec=FAINT, lw=0.55)
        C.text(xb + 1.1, yp + 1.1, h, fontsize=5.1, fontweight='bold', color=SLATE)
        C.text(xb + 1.1, yp + 3.7, t, fontsize=5.0, color=INK, linespacing=1.1)
        if i < 3:
            chevron(C, xb + bw + gap / 2 + 0.45, yp + 4.5, 2.0, w=0.8, color=LIGHT, lw=0.6)


# ======================================================================================== c
def pc(C):
    S = rd('Fig1c_stage_profile.csv')
    kicker(C, 4.2, 0.5, 'Clinical framework')
    title(C, 4.2, 3.3, 'CKM stages 0\u20133 (baseline-verified)')
    cx0, cw = 12.2, 10.5
    ybase, hmax = 31.0, 18.0
    defs = ['No risk\nfactors', 'Excess\nadiposity', 'Metabolic\nrisk factors\nor CKD',
            'Very-high-\nrisk CKD or\nPREVENT\n\u226520%']
    for k, r in S.iterrows():
        cx = cx0 + cw * k + cw / 2
        h = r.pct / 45 * hmax
        C.rect(cx - 3.6, ybase - h, 7.2, h, fc=STAGE_COL[k], ec='none')
        C.text(cx, ybase - h - 3.9, f"{int(r.n):,}", fontsize=5.4, fontweight='bold', ha='center')
        C.text(cx, ybase - h - 1.6, f"{r.pct:.1f}%", fontsize=5.0, ha='center', color=MID)
        C.text(cx, ybase + 1.0, f'Stage {k}', fontsize=5.4, fontweight='bold', ha='center', color=STAGE_COL[k]
               if k else '#6F7F8F')
        C.text(cx, ybase + 3.5, defs[k], fontsize=5.0, ha='center', color=INK, linespacing=1.08)
    C.line([cx0, cx0 + 4 * cw], [ybase, ybase], color=INK, lw=0.5)
    # profile table
    rows = [('Age, y', [f'{v:.1f}' for v in S.age_mean]),
            ('Women, %', [f'{v:.1f}' for v in S.women_pct]),
            ('PREVENT, %', [f'{v:.1f}' for v in S.prevent_median]),
            ('eGFR', [f'{v:.0f}' for v in S.egfr_median])]
    yr = 46.0
    for j, (lab, vals) in enumerate(rows):
        yy = yr + j * 4.2
        if j % 2 == 0:
            C.rect(0.5, yy - 0.75, cx0 + 4 * cw - 0.5, 4.2, fc='#F5F6F7', ec='none', zorder=0)
        C.text(1.0, yy + 0.25, lab, fontsize=5.0, color=MID)
        for k, v in enumerate(vals):
            C.text(cx0 + cw * k + cw / 2, yy + 0.2, v, fontsize=5.1, ha='center',
                   fontweight='bold' if j == 2 else 'normal', color=INK)


# ======================================================================================== d
def pd_(C):
    P = rd('Fig1d_corr_pearson.csv', index_col=0)
    PCA = rd('Fig1d_pca.csv')
    IND = rd('Fig1d_independence.csv')
    PT = rd('Fig1d_pair_types.csv').set_index('pair_type')
    order = list(P.index)
    M = P.to_numpy()
    n = len(order)
    kicker(C, 4.2, 0.5, 'Structure')
    title(C, 4.2, 3.3, 'Organ clocks capture largely distinct ageing axes')
    hx, hy, hs = 15.5, 9.8, 42.0
    cs = hs / n
    ax = C.ax(hx, hy, hs, hs)
    Z = np.full_like(M, np.nan)
    for i in range(n):
        for j in range(i):
            Z[i, j] = M[i, j]
    ax.imshow(Z, cmap=SEQ, vmin=0, vmax=0.5, interpolation='nearest', aspect='auto')
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fc=ORGAN_COL[order[i]], ec='white', lw=0.6))
        for j in range(i):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc='none', ec='white', lw=0.6))
            if M[i, j] >= 0.30:
                ax.text(j, i, f'{M[i, j]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=5.0,
                        color='white')
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, o in enumerate(order):
        C.text(hx - 0.8, hy + cs * (i + 0.5), o, fontsize=5.1, ha='right', va='center', color=ORGAN_COL[o],
               fontweight='bold' if o in TARGET else 'normal')
        C.text(hx + cs * (i + 0.5), hy + hs + 0.8, o, fontsize=5.1, ha='center', va='top', rotation=90,
               color=ORGAN_COL[o], fontweight='bold' if o in TARGET else 'normal')
    # colour bar and summary in the empty upper triangle
    hcolorbar(C, SEQ, Normalize(0, 0.5), hx + 22.5, hy + 3.2, 18.5, 1.3, 'Pearson r', ticks=[0, 0.25, 0.5],
              fmt=lambda t: f'{t:.2f}')
    oo, ow = PT.loc['organ-organ'], PT.loc['organ-whole-body']
    ia = IND[IND['sample'].str.startswith('All (')].set_index('metric')
    i0 = IND[IND['sample'].str.startswith('Stage 0')].set_index('metric')
    txt = [('Median r', ''),
           ('all pairs', f"{ia.loc['r_median', 'estimate']:.2f}"),
           ('stage 0 only', f"{i0.loc['r_median', 'estimate']:.2f}")]
    yy = hy + 9.6
    for k, (a_, b_) in enumerate(txt):
        C.text(hx + 21.5, yy + k * 2.25, a_, fontsize=5.0, fontweight='bold' if k == 0 else 'normal',
               color=INK if k == 0 else MID)
        if b_:
            C.text(hx + hs + 0.2, yy + k * 2.25, b_, fontsize=5.0, ha='right')
    # scree plot
    ax2 = C.ax(66.5, 11.5, 23.5, 17.0)
    clean(ax2, grid='y')
    pcs = PCA.pc.values
    ax2.bar(pcs, PCA.pct, width=0.62, color='#9AA7B4', zorder=2)
    ax2.scatter(pcs, PCA.pct_stage0, s=5, facecolor='white', edgecolor=INK, lw=0.45, zorder=4)
    ax2.plot(pcs, PCA.cum_pct, color=INK, lw=0.8, zorder=3)
    k80 = int(ia.loc['k80', 'estimate'])
    ax2.axhline(80, color=ACCENT, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    ax2.scatter([k80], [PCA.cum_pct.values[k80 - 1]], s=9, color=ACCENT, zorder=5)
    ax2.text(k80 - 0.3, 86, f'80%: {k80} of 12', fontsize=5.0, color=ACCENT, ha='right', va='bottom')
    ax2.text(1.75, 16, f"PC1 = {PCA.pct.values[0]:.1f}%", fontsize=5.0, va='bottom')
    ax2.set_ylim(0, 100); ax2.set_xlim(0.3, 12.7)
    ax2.set_xticks([1, 4, 8, 12]); ax2.set_yticks([0, 50, 100])
    ax2.set_xlabel('Principal component', fontsize=5.2, labelpad=1)
    ax2.set_ylabel('Variance (%)', fontsize=5.2, labelpad=1)
    ax2.tick_params(labelsize=5.0)
    C.text(64.5, 8.4, 'Variance explained', fontsize=5.2, fontweight='bold')
    # effective number of independent clocks
    ax3 = C.ax(66.5, 41.0, 23.5, 15.5)
    clean(ax3, left=False, grid='x')
    items = [('Clocks', 12, 12), ('Meff, Nyholt', ia.loc['meff_nyholt', 'estimate'], i0.loc['meff_nyholt', 'estimate']),
             ('Meff, Li\u2013Ji', ia.loc['meff_liji', 'estimate'], i0.loc['meff_liji', 'estimate']),
             ('PCs for 80%', ia.loc['k80', 'estimate'], i0.loc['k80', 'estimate'])]
    for k, (lab, v, v0) in enumerate(items):
        col = MID if k == 0 else SLATE
        ax3.barh(k, v, height=0.36, color=lighten(col, 0.25) if k else FAINT, zorder=2)
        ax3.scatter([v0], [k], s=7, facecolor='white', edgecolor=INK, lw=0.45, zorder=4)
        ax3.text(0.15, k - 0.3, lab, fontsize=5.0, va='bottom', ha='left', color=INK)
        ax3.text(v + 0.75, k, f'{v:.1f}' if abs(v - round(v)) > 1e-6 else f'{v:.0f}', fontsize=5.0,
                 va='center', ha='left', fontweight='bold')
    ax3.set_ylim(3.6, -0.75); ax3.set_xlim(0, 14.8)
    ax3.set_yticks([]); ax3.set_xticks([0, 4, 8, 12])
    ax3.tick_params(labelsize=5.0)
    ax3.set_xlabel('Number of dimensions', fontsize=5.2, labelpad=1)
    C.text(64.5, 37.4, 'Effective number of clocks', fontsize=5.2, fontweight='bold')
    # shared marker legend (replaces the explanatory footnote)
    C.rect(64.6, 63.45, 2.4, 1.5, fc='#9AA7B4', ec='none')
    C.text(67.8, 64.2, 'All participants', fontsize=5.0, color=MID, va='center')
    circle(C, 81.4, 64.2, 0.62, fc='white', ec=INK, lw=0.45)
    C.text(82.8, 64.2, 'Stage 0', fontsize=5.0, color=MID, va='center')


# ======================================================================================== e
SYS_ICON = {'Cardiac': 'heart-organ', 'Vascular': 'blood-vessel', 'Kidney': 'kidneys', 'Metabolic': 'pancreas',
            'Liver': 'liver', 'Brain': 'neurology', 'Lung': 'lungs', 'Immune/infection': 'bacteria',
            'Musculoskeletal': 'joints', 'Gastrointestinal': 'intestine', 'Eye': 'eye',
            'Cancer': 'cancerous-cell_nuclei'}
SYS_COL = {'Cardiac': 'Heart', 'Vascular': 'Artery', 'Kidney': 'Kidney', 'Metabolic': 'Pancreas', 'Liver': 'Liver',
           'Brain': 'Brain', 'Lung': 'Lung', 'Immune/infection': 'Immune', 'Musculoskeletal': 'Muscle',
           'Gastrointestinal': 'Intestine'}


def pe(C):
    T = rd('Fig1e_tier12_events.csv')
    T3 = rd('Fig1e_tier3_events.csv')
    IM = rd('Fig1e_imaging_phenotypes.csv').set_index('source')
    A = rd('Fig1a_cohort_data_layers.csv').set_index('key').value
    kicker(C, 4.2, 0.5, 'Outcomes')
    title(C, 4.2, 3.3, 'A prespecified three-tier, multi-organ outcome system')
    sysorder = [s for s in ['Cardiac', 'Vascular', 'Kidney', 'Metabolic', 'Liver', 'Brain', 'Lung',
                            'Immune/infection', 'Musculoskeletal', 'Gastrointestinal', 'Eye', 'Cancer']
                if s in set(T.system)]
    others = [s for s in T.system.unique() if s not in sysorder and s != 'Negative control']
    sysorder += others
    rows = sysorder + ['Negative control']
    ax_x, ax_y, ax_w, ax_h = 25.5, 10.0, 35.0, 43.0
    ax = C.ax(ax_x, ax_y, ax_w, ax_h)
    ax.set_xscale('log'); ax.set_xlim(60, 12000); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis='x', which='major', color=FAINT, lw=0.35)
    ax.set_axisbelow(True)
    ax.set_xticks([100, 300, 1000, 3000, 10000]); ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0)
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_visible(False)
    rh = ax_h / len(rows)
    for i, s in enumerate(rows):
        g = T[T.system == s]
        if s == 'Negative control':
            ax.scatter(g.n_events, [i] * len(g), marker='^', s=9, facecolor='white', edgecolor=INK, lw=0.5, zorder=4)
        else:
            t2 = g[g.tier1 == 0]
            ax.scatter(t2.n_events, [i] * len(t2), s=6.5, color='#B9B9B9', edgecolor='white', lw=0.3, zorder=3)
            t1 = g[g.tier1 == 1]
            col = ORGAN_COL.get(SYS_COL.get(s, ''), MID)
            for _, r in t1.iterrows():
                ax.scatter([r.n_events], [i], s=17, color=col, edgecolor=INK, lw=0.4, zorder=5)
                ax.annotate(f'{r.key}', (r.n_events, i), xytext=(3.2, 0), textcoords='offset points',
                            fontsize=5.0, fontweight='bold', color=col, va='center',
                            path_effects=[pe_.withStroke(linewidth=1.6, foreground='white')], zorder=6)
        yy = ax_y + rh * (i + 0.5)
        col = ORGAN_COL.get(SYS_COL.get(s, ''), MID)
        if s in SYS_ICON:
            icon(C, SYS_ICON[s], 1.0, yy - 1.35, 2.7, color=col)
        C.text(4.6, yy, s, fontsize=5.1, va='center', color=INK if s != 'Negative control' else MID)
    # tier 3 strip
    ax3 = C.ax(ax_x, 54.6, ax_w, 3.4)
    rng = np.random.default_rng(3)
    ax3.scatter(T3.n_events, rng.uniform(-0.38, 0.38, len(T3)), s=1.3, color='#8C8C8C', alpha=0.6, lw=0, zorder=3)
    ax3.set_xscale('log'); ax3.set_xlim(60, 12000); ax3.set_ylim(-0.5, 0.5)
    ax3.grid(axis='x', which='major', color=FAINT, lw=0.35); ax3.set_axisbelow(True)
    ax3.set_yticks([]); ax3.spines['left'].set_visible(False)
    ax3.xaxis.set_minor_locator(NullLocator())
    ax3.set_xticks([100, 300, 1000, 3000, 10000]); ax3.set_xticklabels(['100', '300', '1,000', '3,000', '10,000'])
    ax3.tick_params(labelsize=5.0)
    ax3.set_xlabel('Incident events (n)', fontsize=5.2, labelpad=1)
    C.line([1.0, ax_x + ax_w], [54.0, 54.0], color=FAINT, lw=0.4)
    C.text(4.6, 56.3, f'Tier 3 ({len(T3)} phecodes)', fontsize=5.1, va='center')
    # legend
    yl = 65.0
    from matplotlib.lines import Line2D
    lx = 1.0
    for mk, fc, ec, sz, lab in [('o', '#B23A48', INK, 2.2, 'Tier 1'), ('o', '#B9B9B9', 'white', 1.8, 'Tier 2'),
                                ('o', '#8C8C8C', 'none', 1.0, 'Tier 3'), ('^', 'white', INK, 1.9, 'Negative control')]:
        axl = C.ax(lx, yl - 1.2, 2.4, 2.4)
        axl.axis('off'); axl.set_xlim(-1, 1); axl.set_ylim(-1, 1)
        axl.scatter([0], [0], marker=mk, s=sz ** 2 * 3.2, facecolor=fc, edgecolor=ec, lw=0.4)
        C.text(lx + 2.6, yl, lab, fontsize=5.0, va='center')
        lx += 4.0 + len(lab) * 0.95
    # tier hierarchy (right)
    xr = 63.5
    C.text(xr, 9.0, 'Hierarchy of tests', fontsize=5.2, fontweight='bold')
    tiers = [('Tier 1', '6 organ-paired diseases', 'Holm', 13.0), ('Tier 2', '46 diseases, 12 systems', 'FDR', 18.0),
             ('Tier 3', f'{len(T3)} phecodes', 'FDR', 23.5)]
    yt = 12.0
    for k, (t, d, corr, w) in enumerate(tiers):
        xx = xr + (23.5 - w) / 2
        rrect(C, xx, yt, w, 3.0, r=0.8, fc=lighten(SLATE, 0.15 + 0.3 * k), ec='none')
        C.text(xx + w / 2, yt + 0.5, f'{t} \u00b7 {corr}', fontsize=5.0, fontweight='bold',
               color='white' if k < 2 else INK, ha='center')
        C.text(xr + 11.75, yt + 3.35, d, fontsize=5.0, color=MID, ha='center')
        yt += 6.6
    # imaging
    yi = 33.4
    C.text(xr, yi, 'Structure and function', fontsize=5.2, fontweight='bold')
    ims = [('heart-organ', 'Heart', 'Cardiac MRI', f"{int(IM.loc['CMR', 'n_phenotypes'])} phenotypes"),
           ('neurology', 'Brain', 'Brain MRI', f"{int(IM.loc['Brain', 'n_phenotypes'])} phenotypes"),
           ('blood-vessel', 'Artery', 'Arterial stiffness', 'ASI, aortic MRI')]
    yy = yi + 2.8
    for k, (ic, oc, a_, b_) in enumerate(ims):
        rrect(C, xr, yy, 23.5, 6.0, r=1.0, fc='#F5F6F7', ec='none')
        icon(C, ic, xr + 1.0, yy + 1.0, 4.0, color=ORGAN_COL[oc])
        C.text(xr + 5.8, yy + 0.9, a_, fontsize=5.0, fontweight='bold')
        C.text(xr + 5.8, yy + 3.15, b_, fontsize=5.0, color=MID)
        yy += 7.0


# ======================================================================================== f
EVCOL = {'Fig. 2': '#5B7083', 'Fig. 3': BLUE, 'Fig. 4': '#2A7F75', 'Fig. 5': '#6D597A', 'Fig. 6': ACCENT}
BURC = ['#B4BAC1', lighten(ACCENT, 0.62), lighten(ACCENT, 0.3), ACCENT]   # 0, 1, 2, >=3 aged organs
FS_T = 5.0                                                                  # thumbnail text size (pt)
ANS_WRAP = 41                                                               # conclusion line length (chars)


def _thumb(ax, xlabel=None, ylabel=None):
    clean(ax)
    ax.tick_params(labelsize=FS_T, length=1.6, pad=1.0, width=0.4)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_linewidth(0.45)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_T, labelpad=1.2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_T, labelpad=1.2)


def _sup(p):
    """P value as m.m×10^e with Unicode superscripts (no mathtext)."""
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    return f'{m:.1f}\u00d710$^{{\u2212{abs(e)}}}$'


def _q1(ax):
    D = rd('Fig1f_q1_stage_differences.csv')
    T = rd('Fig1f_q1_asynchrony_test.csv').iloc[0]
    late = [c.strip() for c in T.late_rising_clocks.split(',')]
    ax.axhline(0, color=LIGHT, lw=0.45, ls=(0, (2, 1.6)), zorder=1)
    for o in ['Artery', 'Liver', 'Heart', 'Pancreas', 'Brain', 'Kidney']:
        d = D[D.clock == o].sort_values('stage')
        k = o == 'Kidney'
        ax.plot(d.stage, d.beta_sd, color=ORGAN_COL[o], lw=1.15 if k else 0.65, zorder=4 if k else 3,
                marker='o', ms=1.9 if k else 1.3, mec='white', mew=0.3, solid_capstyle='round')
    ax.set_xlim(-0.25, 3.25); ax.set_ylim(-0.66, 1.0)
    ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0', '0.5', '1.0'])
    kid = D[(D.clock == 'Kidney') & (D.stage == 2)].beta_sd.iloc[0]
    ax.text(1.9, kid + 0.1, 'Kidney', fontsize=FS_T, color=ORGAN_COL['Kidney'], ha='right', va='bottom',
            fontweight='bold')
    ax.text(3.2, -0.63, ', '.join(late).replace(', Brain', ',\nbrain').replace('Pancreas', 'pancreas') +
            ': stage 3', fontsize=FS_T, color=MID, ha='right', va='bottom', linespacing=1.05)
    _thumb(ax, 'CKM stage', '\u0394 vs stage 0 (s.d.)')
    return T


def _q2(ax):
    D = rd('Fig1f_q2_specificity.csv')
    order = ['HF', 'CKD', 'DEM', 'ASCVD', 'T2D', 'CLD']
    lab = {'HF': 'HF', 'CKD': 'CKD', 'DEM': 'Dementia', 'ASCVD': 'ASCVD', 'T2D': 'T2D', 'CLD': 'CLD'}
    ranks = {}
    for i, o in enumerate(order):
        d = D[D.outcome == o]
        u, p_ = d[~d.paired], d[d.paired].iloc[0]
        ax.plot([d.hr.min(), d.hr.max()], [i, i], color=FAINT, lw=0.5, zorder=1, solid_capstyle='round')
        ax.scatter(u.hr, [i] * len(u), s=3.2, color=LIGHT, lw=0, zorder=2)
        ax.scatter([p_.hr], [i], s=13, color=ORGAN_COL[p_.clock], edgecolor='white', lw=0.35, zorder=3)
        ranks[o] = int(p_.rank_among_10)
        r = ranks[o]
        ax.text(2.42, i, f'{r}', fontsize=FS_T, ha='right', va='center', color=INK if r == 1 else MID,
                fontweight='bold' if r == 1 else 'normal', clip_on=False)
    ax.axhline(2.5, color=LIGHT, lw=0.4, ls=(0, (2, 1.6)))
    ax.text(2.42, -0.95, 'rank', fontsize=FS_T, ha='right', va='center', color=MID, clip_on=False)
    ax.set_xscale('log'); ax.set_xlim(1.0, 2.15)
    ax.set_xticks([1.0, 1.5, 2.0]); ax.set_xticklabels(['1', '1.5', '2']); ax.minorticks_off()
    ax.set_ylim(5.6, -0.6); ax.set_yticks(range(6)); ax.set_yticklabels([lab[o] for o in order])
    ax.tick_params(axis='y', length=0)
    _thumb(ax, 'HR per s.d. (\u25cf paired clock)')
    ax.spines['left'].set_visible(False)
    return ranks


def _q3(ax):
    D = rd('Fig1f_q3_risk_stage_burden.csv')
    E = rd('Fig1f_q3_equivalence.csv').iloc[0]
    groups, nb = ['0', '1', '2-3'], ['0', '1', '2', '>=3']
    off = [-0.27, -0.09, 0.09, 0.27]
    ref = D[(D.stage.astype(str) == '2-3') & (D.n_aged.astype(str) == '0')].risk10.iloc[0] * 100
    ax.axhline(ref, color=MID, lw=0.45, ls=(0, (2, 1.6)), zorder=1)
    for g_, g in enumerate(groups):
        for b_, b in enumerate(nb):
            r = D[(D.stage.astype(str) == g) & (D.n_aged.astype(str) == b)].iloc[0]
            x = g_ + off[b_]
            ax.plot([x, x], [r.lo * 100, r.hi * 100], color=BURC[b_], lw=0.6, zorder=2)
            ax.scatter([x], [r.risk10 * 100], s=6.5, color=BURC[b_], edgecolor='white', lw=0.3, zorder=3)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(4, 31)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['0', '1', '2\u20133'])
    ax.set_yticks([10, 20, 30])
    xs = -0.43
    ax.text(xs, 30.8, 'Aged organs', fontsize=FS_T, color=MID, ha='left', va='top')
    for b_, s_ in enumerate(['0', '1', '2', '\u22653']):
        ax.text(xs + 1.38 + b_ * 0.24, 30.8, s_, fontsize=FS_T, color=BURC[b_] if b_ else '#8C939B', ha='left',
                va='top', fontweight='bold')
    ax.plot([xs + 0.02, xs + 0.3], [26.6, 26.6], color=MID, lw=0.45, ls=(0, (2, 1.6)))
    ax.text(xs + 0.36, 26.6, 'stage 2\u20133, none aged', fontsize=FS_T, color=MID, ha='left', va='center')
    _thumb(ax, 'CKM stage', '10-y CKM risk (%)')
    return ref, E


def _q4(ax):
    D = rd('Fig1f_q4_delta_c.csv')
    order = ['CKMC', 'HF', 'ASCVD', 'CKD', 'T2D', 'CLD', 'DEM']
    lab = {'CKMC': 'Composite', 'HF': 'HF', 'ASCVD': 'ASCVD', 'CKD': 'CKD', 'T2D': 'T2D', 'CLD': 'CLD',
           'DEM': 'Dementia'}
    ax.axvline(0, color=LIGHT, lw=0.45, zorder=1)
    for i, o in enumerate(order):
        for base, dy, filled in [('Model G', -0.17, True), ('Model D', 0.17, False)]:
            r = D[(D.outcome == o) & (D.baseline == base)].iloc[0]
            col = ORGAN_COL['Brain'] if o == 'DEM' else INK
            ci(ax, r.dC, i + dy, r.lo, r.hi, color=col, ms=1.75, lw=0.55, filled=filled)
    ax.set_xlim(-0.006, 0.034); ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xticklabels(['0', '0.01', '0.02', '0.03'])
    ax.set_ylim(6.6, -0.6); ax.set_yticks(range(7)); ax.set_yticklabels([lab[o] for o in order])
    ax.tick_params(axis='y', length=0)
    _thumb(ax, '\u0394C (\u25cf over G, \u25cb over D)')
    ax.spines['left'].set_visible(False)
    return D


def _q5(ax):
    D = rd('Fig1f_q5_observed_predicted.csv').sort_values('bcat')
    lab = ['None aged', '1 aged', '\u22652 aged']
    cols = ['#8C939B', lighten(ACCENT, 0.45), ACCENT]
    for i, r in enumerate(D.itertuples()):
        ax.plot([r.pred, r.obs], [i, i], color=lighten(cols[i], 0.45), lw=1.6, zorder=1, solid_capstyle='butt')
        ax.scatter([r.pred], [i], s=11, facecolor='white', edgecolor=cols[i], lw=0.7, zorder=3)
        ax.scatter([r.obs], [i], s=11, color=cols[i], edgecolor='white', lw=0.3, zorder=3)
        if i == 2:
            ax.text(r.pred + 0.6, i + 0.3, 'PREVENT', fontsize=FS_T, ha='right', va='top', color=MID)
            ax.text(r.obs - 0.6, i + 0.3, 'observed', fontsize=FS_T, ha='left', va='top', color=ACCENT)
        ax.text(19.6, i, f'{r.oe:.2f}', fontsize=FS_T, ha='right', va='center',
                color=cols[i] if i else MID, fontweight='bold' if i == 2 else 'normal')
    ax.text(19.6, -0.8, 'O/E', fontsize=FS_T, ha='right', va='center', color=MID)
    ax.set_xlim(4, 20); ax.set_xticks([5, 10, 15])
    ax.set_ylim(3.1, -1.0); ax.set_yticks(range(3)); ax.set_yticklabels(lab)
    ax.tick_params(axis='y', length=0)
    _thumb(ax, '10-y CVD risk (%)')
    ax.spines['left'].set_visible(False)
    return D


def pf(C):
    kicker(C, 4.2, 0.5, 'Analytical roadmap')
    title(C, 4.2, 3.3, 'Five sequential questions: each answer is the premise of the next')
    cw, x00 = 36.0, 1.0
    X0 = [x00 + k * cw for k in range(5)]
    figs = ['Fig. 2', 'Fig. 3', 'Fig. 4', 'Fig. 5', 'Fig. 6']
    level = ['Association', 'Organ specificity', 'Stratification beyond stage', 'Incremental prediction',
             'Decision relevance']
    quest = ['Do organs age in step with CKM stage?',
             'Does each clock predict disease of its own organ?',
             'Does ageing burden reorder risk within a stage?',
             'Do age gaps add to guideline-based risk models?',
             'Where could organ ageing inform a decision?']
    premise = ['asynchronous', 'specific in 3 of 6 pairs', 'stratifies within stage', 'small \u0394C: calibration?']
    # ---- logic spine
    ys = 11.2
    for k in range(5):
        xn = X0[k] + 2.6
        circle(C, xn, ys, 1.45, fc=EVCOL[figs[k]], ec='none', zorder=6)
        C.text(xn, ys + 0.05, f'{k + 1}', fontsize=5.2, color='white', ha='center', va='center',
               fontweight='bold', zorder=7)
        if k < 4:
            C.arrow(xn + 1.75, ys, X0[k + 1] + 2.6 - 1.75, ys, color=LIGHT, lw=0.5,
                    arrowstyle='-|>,head_length=1.6,head_width=0.8')
            C.text(X0[k + 1] + 0.2, ys - 0.75, premise[k], fontsize=5.0, color=MID, ha='right', va='bottom',
                   style='italic')
    C.line([X0[4] + 2.6 + 1.75, 178.8], [ys, ys], color=LIGHT, lw=0.5, ls=(0, (2, 1.6)))
    C.text(178.8, ys - 0.75, 'prospective evaluation', fontsize=5.0, color=MID, ha='right', va='bottom',
           style='italic')
    # ---- column separators
    for k in range(1, 5):
        C.line([X0[k] - 1.2] * 2, [14.6, 53.4], color=FAINT, lw=0.35)
    # ---- thumbnails (axes boxes: x offset, width)
    geo = [(6.8, 26.0), (8.6, 21.6), (6.8, 26.0), (9.2, 23.6), (8.9, 21.6)]
    yt, ht = 24.6, 17.4
    axs = [C.ax(X0[k] + geo[k][0], yt, geo[k][1], ht) for k in range(5)]
    T1 = _q1(axs[0]); R2 = _q2(axs[1]); ref, E3 = _q3(axs[2]); D4 = _q4(axs[3]); D5 = _q5(axs[4])
    # ---- answers and design lines (numbers formatted from the source data)
    g4 = D4[D4.baseline == 'Model G'].set_index('outcome')
    others = g4.drop(index='DEM')
    ns = [o for o in g4.index if g4.loc[o, 'lo'] <= 0]
    assert ns == ['CKD'], ns
    o5 = D5.set_index('bcat')
    spec = [o for o, r in R2.items() if r == 1]
    assert spec == ['HF', 'CKD', 'DEM'], spec
    answers = [
        'Graded but asynchronous: kidney, artery and liver rise from stage 1.',
        f'Ranks 1st for HF, CKD and dementia; {R2["ASCVD"]}th, {R2["T2D"]}th and {R2["CLD"]}th for ASCVD, T2D '
        f'and CLD.',
        f'Risk rises with burden; stage 0\u20131 with \u22652 aged \u2248 stage 2\u20133 with none '
        f'(HR\u00a0{E3.hr_LH_vs_HL:.2f}).',
        f'Consistent but modest (\u0394C\u00a0\u2264\u00a0{others.dC.max():.3f}; dementia '
        f'{g4.loc["DEM", "dC"]:.3f}); CKD not significant.',
        f'PREVENT under-predicts with \u22652 aged organs (O/E\u00a0{o5.loc[2, "oe"]:.2f}); proposed, not tested.',
    ]
    for k in range(5):
        assert len(textwrap.wrap(answers[k], ANS_WRAP)) <= 2, (k, textwrap.wrap(answers[k], ANS_WRAP))
        x = X0[k] + 0.6
        C.text(x, 14.6, figs[k], fontsize=5.6, fontweight='bold', color=EVCOL[figs[k]])
        C.text(x + 7.0, 14.75, level[k], fontsize=5.0, color=MID)
        C.text(x, 17.6, wrap(quest[k], 34), fontsize=5.5, fontweight='bold', linespacing=1.1)
        C.text(x, 48.0, wrap(answers[k], ANS_WRAP), fontsize=5.1, color=INK, linespacing=1.12)


# ======================================================================================== layout
W = 180.0
PANELS = {
    'a': (pa, 0.0, 0.0, 60.0, 66.0),
    'b': (pb, 61.5, 0.0, 62.0, 66.0),
    'c': (pc, 125.0, 0.0, 55.0, 66.0),
    'd': (pd_, 0.0, 69.0, 91.0, 68.0),
    'e': (pe, 92.5, 69.0, 87.5, 68.0),
    'f': (pf, 0.0, 140.0, 180.0, 54.5),
}
H = 194.5

if __name__ == '__main__':
    sel = sys.argv[1:]
    if sel:
        for lab in sel:
            fn, x, y, w, h = PANELS[lab]
            single(fn, w, h, f'Fig1{lab}', os.path.join(ROOT, 'panels'), lab=lab)
    else:
        build(PANELS, W, H, 'Fig1', ROOT)
