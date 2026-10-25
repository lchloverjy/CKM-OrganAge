"""Figure 6 (v2). Organ age along the guideline Calculate-Personalize-Reclassify pathway.
Reads only ../source_data/*.csv.   Run:  cd code && python fig6v2.py
Each panel function draws into a millimetre Canvas; the same function is used for the single-panel export and the
composite, so panels/Fig6x.* are identical to their position in Fig6.*.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style4 import *
import pandas as pd, numpy as np
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import FancyBboxPatch
from scipy.stats import chi2
SD = os.path.join(HERE, '..', 'source_data') + '/'
ROOT = os.path.join(HERE, '..')
rd = lambda n: pd.read_csv(SD + n + '.csv')
B3C = ['#9C9C9C', '#E39B5B', '#A4161A']
B3L = ['0', '1', '≥2']
STEPC = '#2F5D8C'
RANKC = {'Rank by PREVENT': '#2F5D8C', 'Rank by age': '#8A7FA0'}


def head(C, step, title, x=4.3):
    C.text(x, 0.55, step, fontsize=5.0, color=STEPC, fontweight='bold', va='top')
    C.text(x, 3.15, title, fontsize=6.5, fontweight='bold', va='top')


def burden_legend(C, x, y, title='Aged target organs'):
    C.text(x, y, title, fontsize=5.3, color=MID, va='center')
    xx = x + len(title) * 0.98 + 1.6
    for b in range(3):
        ax = C.ax(xx - 0.9, y - 0.9, 1.8, 1.8); ax.set_axis_off(); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.scatter([0], [0], s=7, color=B3C[b], lw=0)
        C.text(xx + 1.3, y, B3L[b], fontsize=5.3, va='center')
        xx += 6.0


def fci(h, l, u, nd=2):
    return um(f'{h:.{nd}f} ({l:.{nd}f}–{u:.{nd}f})')


def sqrt_axis(ax, which):
    f = (lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: np.square(v))
    (ax.set_xscale if which == 'x' else ax.set_yscale)('function', functions=f)


# ============================================================================================== a  CALCULATE
def pa(C):
    head(C, 'CALCULATE', 'PREVENT under-predicts with organ ageing')
    burden_legend(C, 4.3, 9.3)
    D = rd('Fig6a_calibration_deciles'); OE = rd('Fig6a_oe_by_burden').set_index('bcat')
    ax = C.ax(11.5, 12.5, 43.0, 37.5)
    sqrt_axis(ax, 'x'); sqrt_axis(ax, 'y')
    ax.set_xlim(0.6, 19); ax.set_ylim(0.3, 33)
    # guideline tiers (x) shaded, thresholds (y) dashed
    for lo_, hi_ in [(5, 7.5)]:
        ax.axvspan(lo_, hi_, color=BAND, lw=0, zorder=0)
    for th in [5, 7.5]:
        ax.axvline(th, color=FAINT, lw=0.45, zorder=0)
    for th, lab in [(5, '5%'), (7.5, '7.5%'), (20, '20%')]:
        ax.axhline(th, color=LIGHT, lw=0.45, ls=(0, (3, 2)), zorder=0)
        ax.text(19.6, th, lab, fontsize=5.0, color=MID, va='center', ha='left', clip_on=False)
    for x_, lab in [(2.2, '<5%'), (6.1, '5–<7.5'), (12.0, '7.5–<20')]:
        ax.text(x_, 31.5, lab, fontsize=5.0, color=MID, ha='center', va='top')
    xx = np.linspace(0.6, 19, 50); ax.plot(xx, xx, color=INK, lw=0.5, ls=(0, (1.2, 1.2)), zorder=1)
    ax.text(0.97, 0.035, 'dotted line, perfect calibration', transform=ax.transAxes, fontsize=5.0, color=MID, ha='right', va='bottom')
    for b in [0, 1, 2]:
        g = D[D.bcat == b].sort_values('decile')
        if b != 1:
            ax.fill_between(g.pred, g.lo, g.hi, color=B3C[b], alpha=0.16 if b == 2 else 0.22, lw=0, zorder=2)
        ax.plot(g.pred, g.obs, color=B3C[b], lw=0.8, zorder=3)
        ax.scatter(g.pred, g.obs, s=7, color=B3C[b], edgecolor='white', linewidth=0.3, zorder=4)
    ax.set_xticks([1, 2, 5, 10, 15]); ax.set_xticklabels(['1', '2', '5', '10', '15'])
    ax.set_yticks([1, 2, 5, 10, 20, 30]); ax.set_yticklabels(['1', '2', '5', '10', '20', '30'])
    ax.set_xlabel('PREVENT-predicted 10-y total CVD (%)'); ax.set_ylabel('Observed 10-y total CVD (%)')
    clean(ax)
    ax.text(0.03, 0.80, 'Observed / expected', transform=ax.transAxes, fontsize=5.0, color=MID, va='top')
    for i, b in enumerate([2, 1, 0]):
        r = OE.loc[b]
        ax.text(0.03, 0.72 - i * 0.075, f'{B3L[b]}:', transform=ax.transAxes, fontsize=5.1, color=B3C[b], va='top', fontweight='bold')
        ax.text(0.115, 0.72 - i * 0.075, fci(r.oe, r.oe_lo, r.oe_hi), transform=ax.transAxes, fontsize=5.1, color=B3C[b], va='top')
    H = rd('Fig6a_hr_overall'); h = H[(H.outcome == 'CVD') & (H.term == 'b2')].iloc[0]
    S = rd('Fig6a_sensitivity_biomarkers'); g = S[(S.outcome == 'CVD') & (S.term == 'b2') & (S.adjustment == '+ NT-proBNP + troponin I + eGFRcr-cys + UACR')].iloc[0]
    C.text(4.3, 56.6, f'≥2 vs 0 aged organs, HR {fci(h.hr, h.lo, h.hi)}', fontsize=5.0, color=MID)
    C.text(4.3, 58.9, f'+ NT-proBNP, troponin I, eGFRcr-cys, UACR: {fci(g.hr, g.lo, g.hi)}', fontsize=5.0, color=MID)


# ============================================================================================== b  PERSONALIZE (thresholds)
def pb(C):
    head(C, 'PERSONALIZE', 'Organ ageing and guideline action thresholds')
    burden_legend(C, 4.3, 9.3)
    T = rd('Fig6b_threshold_upgrade')
    tiers = [('<5%', 5.0, 'PREVENT <5%'), ('5-<7.5%', 7.5, '5–<7.5%'), ('7.5-<20%', 20.0, '7.5–<20%')]
    ax = C.ax(15.5, 13.0, 26.5, 18.0); ax2 = C.ax(45.0, 13.0, 12.5, 18.0)
    ax.set_xscale('log'); ax.set_xlim(1.8, 30); ax.set_ylim(-0.6, 2.6)
    for i, (tier, pt, lab) in enumerate(tiers):
        y = 2 - i
        ax.add_patch(Rectangle((pt, y - 0.42), 40 - pt, 0.84, fc=lighten(B3C[2], 0.9), ec='none', zorder=0))
        ax.plot([pt, pt], [y - 0.42, y + 0.42], color=B3C[2], lw=0.6, ls=(0, (2, 1.2)), zorder=1)
        ax.text(pt * 1.04, y + 0.40, f'{pt:g}%', fontsize=5.0, color=B3C[2], va='top', ha='left')
        g = T[T.tier == tier].set_index('bcat')
        ax.plot([g.loc[0, 'obs'], g.loc[2, 'obs']], [y + 0.2, y - 0.2], color=FAINT, lw=0.6, zorder=1)
        for b, dy in zip([0, 1, 2], [0.2, 0.0, -0.2]):
            r = g.loc[b]
            ci(ax, np.array([r.obs]), np.array([y + dy]), np.array([r.lo]), np.array([r.hi]), color=B3C[b], ms=2.4, lw=0.6)
        C.text(14.8, 13.0 + 18.0 * (2.6 - y) / 3.2, lab, fontsize=5.3, ha='right', va='center')
        r2 = g.loc[2]
        ci(ax2, np.array([r2.dnb_per1000_upgraded]), np.array([y]), np.array([r2.dnb_lo]), np.array([r2.dnb_hi]),
           color=B3C[2], ms=2.4, lw=0.6, filled=r2.dnb_lo > 0)
        ax2.text(62, y + 0.33, um(f'{r2.dnb_per1000_upgraded:+.1f}'), fontsize=5.0, color=B3C[2] if r2.dnb_lo > 0 else MID,
                 ha='right', va='center', fontweight='bold' if r2.dnb_lo > 0 else 'normal')
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks([2, 5, 10, 20]); ax.set_xticklabels(['2', '5', '10', '20']); ax.minorticks_off()
    ax.set_xlabel('Observed 10-y CVD (%)', labelpad=1.5)
    clean(ax, left=False)
    ax2.set_xlim(-30, 62); ax2.set_ylim(-0.6, 2.6); ax2.axvline(0, color=INK, lw=0.45)
    ax2.set_yticks([]); ax2.set_xticks([0, 50]); clean(ax2, left=False)
    ax2.set_xlabel('ΔNB /1,000', labelpad=1.5)
    C.text(51.2, 10.9, 'Upgrade ≥2', fontsize=5.0, color=MID, ha='center')
    # bottom: borderline tier, upgrading by one enhancer
    K = rd('Fig6b_threshold_upgrade_comparators'); K = K[K.tier == '5-<7.5%'].set_index('enhancer')
    order = [('>=2 aged organs', '≥2 aged organs'), ('UACR >=3 mg/mmol', 'UACR ≥3 mg/mmol'), ('Lp(a) >=125 nmol/L', 'Lp(a) ≥125 nmol/L'),
             ('hsCRP >=2 mg/L', 'hsCRP ≥2 mg/L')]
    C.text(4.3, 38.4, 'Borderline 5–<7.5%: upgrade by one enhancer', fontsize=5.4, fontweight='bold')
    ax3 = C.ax(25.5, 45.0, 19.0, 11.0)
    C.text(48.3, 42.0, '% of\ntier', fontsize=5.0, color=MID, ha='center', linespacing=0.95)
    C.text(55.8, 42.0, 'per 1,000\nin tier', fontsize=5.0, color=MID, ha='center', linespacing=0.95)
    for i, (k, lab) in enumerate(order):
        r = K.loc[k]; y = 3 - i; col = B3C[2] if i == 0 else BLUE
        ci(ax3, np.array([r.dnb_per1000_upgraded]), np.array([y]), np.array([r.dnb_lo]), np.array([r.dnb_hi]), color=col, ms=2.3, lw=0.6,
           filled=r.dnb_lo > 0)
        yy = 45.0 + 11.0 * (3.6 - y) / 4.2
        C.text(25.0, yy, lab, fontsize=5.1, ha='right', va='center', color=INK if i == 0 else MID, fontweight='bold' if i == 0 else 'normal')
        C.text(48.3, yy, f'{r.share_of_tier:.1f}', fontsize=5.1, ha='center', va='center')
        C.text(55.8, yy, um(f'{r.dnb_per1000_upgraded * r.share_of_tier / 100:+.1f}'), fontsize=5.1, ha='center', va='center',
               color=B3C[2] if i == 0 else INK, fontweight='bold' if i == 0 else 'normal')
    ax3.set_ylim(-0.6, 3.6); ax3.set_xlim(-20, 120); ax3.axvline(0, color=INK, lw=0.45)
    ax3.set_yticks([]); ax3.set_xticks([0, 50, 100]); clean(ax3, left=False)
    ax3.set_xlabel('ΔNB per 1,000 upgraded', labelpad=0.8)


# ============================================================================================== c  PERSONALIZE (benchmark)
def pc(C):
    head(C, 'PERSONALIZE', 'Benchmark against guideline risk enhancers')
    P = rd('Fig6c_enhancer_paf'); P = P[P.outcome == 'CVD'].set_index('enhancer')
    # label text, anchor (prevalence, HR) in data units, ha
    LAB = {'>=2 aged organs': ('≥2 aged organs', (18.2, 1.93), 'center'), 'hsCRP >=2 mg/L': ('hsCRP ≥2', (41, 1.33), 'center'),
           'Lp(a) >=125 nmol/L': ('Lp(a) ≥125', (9.8, 1.072), 'right'), 'ApoB >=1.3 g/L': ('ApoB ≥1.3', (9.6, 0.925), 'right'),
           'Triglycerides >=1.98 mmol/L': ('TG ≥1.98', (33, 1.062), 'left'), 'UACR >=3 mg/mmol': ('UACR ≥3', (5.9, 1.43), 'left'),
           'eGFR 45-59': ('eGFR 45–59', (0.98, 1.045), 'left'), 'Highest deprivation quintile': ('Deprivation Q5', (24.5, 0.875), 'left'),
           'Prior depression': ('Depression', (1.12, 2.13), 'left'), 'Chronic inflammatory disease': ('Inflammatory\ndisease', (2.35, 1.585), 'left')}
    GC = {'organ age': B3C[2], 'biomarker': BLUE, 'kidney': '#3D5A80', 'social/clinical': '#7A7A7A'}
    sz = lambda paf: 4 + max(paf, 0.15) * 15
    ax = C.ax(11.5, 9.5, 45.0, 29.0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.55, 70); ax.set_ylim(0.82, 2.75)
    ax.axhline(1, color=INK, lw=0.45)
    for k, r in P.iterrows():
        sig = r.lo > 1; col = GC[r.group]
        ax.errorbar(r.prev, r.hr, yerr=[[r.hr - r.lo], [r.hi - r.hr]], fmt='none', ecolor=lighten(col, 0.35), elinewidth=0.5, zorder=2)
        ax.scatter(r.prev, r.hr, s=sz(r.paf), color=col if sig else 'white', edgecolor=col, linewidth=0.6, alpha=0.88 if sig else 1, zorder=3)
        lab, (lx, ly), ha = LAB[k]
        main = k == '>=2 aged organs'
        ax.text(lx, ly, lab, fontsize=5.3 if main else 5.0, ha=ha, va='center', color=INK if main else MID,
                fontweight='bold' if main else 'normal', zorder=5, linespacing=0.95)
        if k in ['Chronic inflammatory disease', 'eGFR 45-59', 'UACR >=3 mg/mmol', 'Highest deprivation quintile', 'Lp(a) >=125 nmol/L']:
            ax.plot([r.prev, lx * (0.97 if ha == 'left' else 1.03)], [r.hr, ly], color=LIGHT, lw=0.35, zorder=1)
    b = P.loc['>=2 aged organs']
    ax.text(18.2, 1.755, um(f'PAF {b.paf:.1f}% ({b.paf_lo:.1f}–{b.paf_hi:.1f})'), fontsize=5.0, color=B3C[2], ha='center', va='center')
    ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(['1', '2', '5', '10', '20', '50'])
    ax.set_yticks([0.9, 1, 1.5, 2, 2.5]); ax.set_yticklabels(['0.9', '1', '1.5', '2', '2.5']); ax.minorticks_off()
    ax.set_xlabel('Prevalence (%)', labelpad=1.0); ax.set_ylabel('Mutually adjusted HR, total CVD')
    clean(ax)
    # PAF size key (top right, inside the plotting area)
    kx = C.ax(38.0, 9.8, 18.0, 5.0); kx.set_axis_off(); kx.set_xlim(0, 18); kx.set_ylim(0, 5)
    kx.text(0.0, 2.5, 'PAF', fontsize=5.0, color=MID, va='center')
    for xv, v, dx in [(5.0, 1, 1.3), (11.5, 5, 2.4)]:
        kx.scatter([xv], [2.5], s=sz(v), color='white', edgecolor=MID, lw=0.5, clip_on=False)
        kx.text(xv + dx, 2.5, f'{v}%', fontsize=5.0, color=MID, va='center')
    # incremental discrimination beyond PREVENT + all 9 available enhancers
    Q = rd('Fig6c_dc_beyond_enhancers'); q = Q[(Q.outcome == 'CVD') & (Q['sample'] == 'complete case')].set_index('model')
    C.text(4.3, 45.4, 'ΔC beyond PREVENT + 9 enhancers (cross-fitted)', fontsize=5.4, fontweight='bold')
    ax3 = C.ax(23.5, 48.3, 27.0, 6.0)
    for i, (m, lab) in enumerate([('+burden', '+ burden (0/1/≥2)'), ('+12 gaps', '+ 12 age gaps')]):
        r = q.loc[m]; y = 1 - i
        ax3.barh(y, r.dc, height=0.55, color=B3C[2] if i == 0 else lighten(B3C[2], 0.45), lw=0)
        ax3.errorbar(r.dc, y, xerr=[[r.dc - r.lo], [r.hi - r.dc]], fmt='none', ecolor=INK, elinewidth=0.5, capsize=0)
        ax3.text(r.hi + 0.0004, y, um(f'+{r.dc:.4f}'), fontsize=5.0, va='center', ha='left')
        C.text(23.0, 48.3 + 6.0 * (1.6 - y) / 2.2, lab, fontsize=5.0, ha='right', va='center')
    ax3.set_ylim(-0.6, 1.6); ax3.set_xlim(0, 0.016); ax3.set_yticks([]); ax3.set_xticks([0, 0.005, 0.010, 0.015])
    ax3.set_xticklabels(['0', '0.005', '0.010', '0.015']); clean(ax3, left=False)
    lr = Q[(Q.outcome == 'CVD') & (Q['sample'] == 'complete case') & (Q.model == 'LR burden')].iloc[0]
    mi = Q[(Q.outcome == 'CVD') & (Q['sample'] == 'missing indicator') & (Q.model == '+burden')].iloc[0]
    C.text(4.3, 58.7, um(f'n = {int(lr.n):,}, {int(lr.events):,} events; LR χ² (burden) = {lr.dc:.0f}, P = ') + fmt_p(lr.hi), fontsize=5.0, color=MID)
    C.text(4.3, 60.9, um(f'Missing-indicator model (n = {int(mi.n):,}): ΔC +{mi.dc:.4f} ({mi.lo:.4f}–{mi.hi:.4f})'), fontsize=5.0, color=MID)


# ============================================================================================== d  RECLASSIFY
def pd_(C):
    head(C, 'RECLASSIFY', 'Organ-specific hidden risk behind a normal routine assessment')
    H = rd('Fig6d_hidden_organ_risk').set_index('organ')
    rows = [('Heart', 'Heart → heart failure', 'PREVENT-HF <5%'), ('Brain', 'Brain → dementia', 'age ≥60 y, no routine test'),
            ('Kidney', 'Kidney → CKD', 'eGFR ≥60 and UACR <3'), ('Artery', 'Artery → ASCVD', 'PREVENT-ASCVD <5%'),
            ('Liver', 'Liver → liver disease', 'FIB-4 <1.3'), ('Pancreas', 'Pancreas → T2D', 'HbA1c <39 mmol/mol')]
    y0, rh = 13.2, 5.75
    ax = C.ax(38.5, y0, 27.5, rh * 6)
    C.text(52.2, 10.3, '10-y incidence (%)', fontsize=5.2, color=MID, ha='center')
    C.text(76.0, 10.3, 'HR (95% CI)', fontsize=5.2, color=MID, ha='center')
    C.text(91.0, 9.0, 'aged share of\npeople → events', fontsize=5.0, color=MID, ha='center', linespacing=0.95)
    for i, (org, lab, cond) in enumerate(rows):
        r = H.loc[org]; y = 5 - i; sig = (r.q < 0.05) and r.hr > 1.5
        col = ORGAN_COL[org] if sig else LIGHT
        if sig:
            ax.axhspan(y - 0.46, y + 0.46, color=BAND, lw=0, zorder=0)
        ax.plot([r.cif_not, r.cif_aged], [y, y], color=lighten(col, 0.3), lw=1.6 if sig else 1.0, solid_capstyle='round', zorder=1)
        ax.errorbar(r.cif_aged, y, xerr=[[r.cif_aged - r.cif_aged_lo], [r.cif_aged_hi - r.cif_aged]], fmt='none', ecolor=col, elinewidth=0.5, zorder=2)
        ax.scatter([r.cif_not], [y], s=11, facecolor='white', edgecolor=col if sig else LIGHT, lw=0.7, zorder=3)
        ax.scatter([r.cif_aged], [y], s=11, color=col, lw=0, zorder=3)
        yy = y0 + rh * (5.5 - y)
        C.text(37.2, yy - 1.1, lab, fontsize=5.6, ha='right', va='center', color=ORGAN_COL[org] if sig else MID, fontweight='bold' if sig else 'normal')
        C.text(37.2, yy + 1.35, f'{cond}; {r.pct_aged:.0f}% aged', fontsize=5.0, ha='right', va='center', color=MID)
        C.text(76.0, yy - 0.6, f'{r.hr:.2f}', fontsize=5.6, ha='center', va='center', fontweight='bold' if sig else 'normal', color=INK if sig else LIGHT)
        C.text(76.0, yy + 1.6, um(f'{r.hr_lo:.2f}–{r.hr_hi:.2f}'), fontsize=5.0, ha='center', va='center', color=MID if sig else LIGHT)
        C.text(91.0, yy, f'{r.pct_aged:.0f}% → {r.share_events_flagged:.0f}%', fontsize=5.3, ha='center', va='center',
               color=ORGAN_COL[org] if sig else LIGHT, fontweight='bold' if sig else 'normal')
    ax.set_ylim(-0.55, 5.55); ax.set_xlim(0, 7.2); ax.set_yticks([]); ax.set_xticks([0, 2, 4, 6])
    clean(ax, left=False, grid='x')
    C.text(4.3, 53.4, 'Open, paired clock not aged; filled, aged (sex- and age-specific top decile). Grey rows: q ≥0.05 or HR <1.5', fontsize=5.0, color=MID)
    # cumulative incidence curves
    K = rd('Fig6d_hidden_curves'); S = rd('Fig6d_hidden_sensitivity')
    spec = [('Heart', 'HF', 'Heart failure', 'PREVENT-HF <5%', 4.0), ('Kidney', 'CKD', 'Chronic kidney disease', 'eGFR ≥60, UACR <3', 8.0),
            ('Brain', 'DEM', 'Dementia', 'age ≥60 y', 9.0)]
    for j, (org, key, ttl, cond, ymax) in enumerate(spec):
        x0 = 103.0 + j * 26.0
        a = C.ax(x0, 15.5, 19.0, 24.5)
        for aged, col in [(0, LIGHT), (1, ORGAN_COL[org])]:
            g = K[(K.organ == org) & (K.aged == aged) & (K.t <= 10.0)].sort_values('t')
            a.fill_between(g.t, g.lo, g.hi, color=col, alpha=0.18, lw=0, step='post')
            a.step(g.t, g.cif, where='post', color=col, lw=0.9 if aged else 0.8)
            v = g.loc[(g.t - 10).abs().idxmin(), 'cif']
            a.text(10.4, v, f'{v:.1f}', fontsize=5.0, color=col, va='center', ha='left', fontweight='bold' if aged else 'normal')
        a.set_xlim(0, 10); a.set_ylim(0, ymax); a.set_xticks([0, 5, 10])
        a.set_xlabel('Years', labelpad=1.0)
        if j == 0:
            a.set_ylabel('Cumulative incidence (%)')
        clean(a)
        C.text(x0, 9.6, ttl, fontsize=5.6, fontweight='bold', color=ORGAN_COL[org])
        C.text(x0, 12.0, cond, fontsize=5.0, color=MID)
        sens = S[(S.organ == org) & ~S.adjustment.str.startswith('main')].iloc[0]
        C.text(x0, 49.2, sens.adjustment.replace('+ ', '+').replace('APOE e4', 'APOE ε4'), fontsize=5.0, color=MID)
        C.text(x0, 51.6, f'HR {fci(sens.hr, sens.lo, sens.hi)}', fontsize=5.0, color=INK)
    C.text(178.0, 9.6, 'aged', fontsize=5.0, color=INK, ha='right', fontweight='bold')
    C.text(178.0, 12.0, 'not aged', fontsize=5.0, color=LIGHT, ha='right')


# ============================================================================================== e  COORDINATE
def pe(C):
    head(C, 'COORDINATE', 'Who needs coordinated multi-organ care')
    G = rd('Fig6e_risk_grid'); DCc = rd('Fig6e_dc_corrected')
    spec = [('OVL', 'New CKM overlap', '(≥2 of T2D, CKD, CVD)', (0, 36)), ('MSYS', '≥2 newly affected', 'organ systems', (10, 60))]
    for j, (key, t1, t2, rng_) in enumerate(spec):
        x0 = 17.0 + j * 41.0
        ax = C.ax(x0, 16.5, 30.0, 22.5)
        g = G[(G.outcome == key) & (G.sg.isin(['0', '1', '2']))].copy(); g['sg'] = g.sg.astype(int); g['bcat'] = g.bcat.astype(int)
        M = np.full((3, 3), np.nan); N = np.zeros((3, 3), int)
        for _, r in g.iterrows():
            M[r.sg, r.bcat] = r.cif10; N[r.sg, r.bcat] = r.n
        norm = Normalize(*rng_)
        ax.imshow(M, cmap=SEQ, norm=norm, aspect='auto')
        for a_ in range(3):
            for b_ in range(3):
                v = M[a_, b_]; dark = norm(v) > 0.55
                ax.text(b_, a_ - 0.12, f'{v:.1f}', ha='center', va='center', fontsize=6.0, fontweight='bold', color='white' if dark else INK)
                ax.text(b_, a_ + 0.27, f'n={N[a_, b_]:,}', ha='center', va='center', fontsize=5.0, color='white' if dark else MID)
        ax.set_xticks(range(3)); ax.set_xticklabels(B3L); ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.2)
        ax.set_yticks(range(3)); ax.set_yticklabels(['Stage 0–1', 'Stage 2', 'Stage 3'] if j == 0 else ['', '', ''])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 3), minor=True); ax.set_yticks(np.arange(-0.5, 3), minor=True)
        ax.grid(which='minor', color='white', lw=1.0); ax.tick_params(which='minor', length=0)
        C.text(x0 + 15, 9.4, t1, fontsize=5.6, fontweight='bold', ha='center')
        C.text(x0 + 15, 11.8, t2, fontsize=5.0, color=MID, ha='center')
        h = G[(G.outcome == key) & (G.bcat == 'HR_b2')].iloc[0]
        dc = DCc[(DCc.outcome == key) & (DCc.model == '+12 gaps')].iloc[0]
        C.text(x0 + 15, 41.0, f'HR ≥2 vs 0: {fci(h.cif10, h.lo, h.hi)}', fontsize=5.0, ha='center')
        C.text(x0 + 15, 43.5, um(f'ΔC +12 gaps: +{dc.dc:.3f} ({dc.lo:.3f}–{dc.hi:.3f})'), fontsize=5.0, ha='center', color=BLUE)
    C.text(16.6, 14.6, 'Aged organs →', fontsize=5.0, color=MID, ha='right')
    C.text(4.3, 48.3, '10-y Aalen–Johansen risk (%), death as competing event. HR adjusted for log PREVENT, stage and', fontsize=5.0, color=MID)
    C.text(4.3, 50.5, 'model a covariates; ΔC, cross-fitted gain of the 12 age gaps over stage + PREVENT + covariates', fontsize=5.0, color=MID)


# ============================================================================================== f  IMPLEMENT
def pf(C):
    head(C, 'IMPLEMENT', 'Whom to test first: efficiency frontier, CKM stages 1–2')
    F = rd('Fig6f_frontier_curves'); NS = rd('Fig6f_named_strategies').set_index('strategy')
    ax = C.ax(12.0, 10.5, 33.5, 32.0)
    ax.plot([0, 100], [0, 100], color=LIGHT, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(25, 12, 'random', fontsize=5.0, color=LIGHT, ha='left')
    for st, col in RANKC.items():
        g = F[F.strategy == st]
        ax.plot(g.pct_tested, g.pct_events, color=col, lw=0.9, zorder=2)
    g = F[F.strategy == 'Rank by PREVENT']
    ax.plot(g.pct_tested, g.pct_upclass, color=RANKC['Rank by PREVENT'], lw=0.7, ls=(0, (2.2, 1.2)), zorder=2)
    ax.text(71, 22, 'flags <7.5%\ncaptured', fontsize=5.0, color=RANKC['Rank by PREVENT'], ha='left', va='center', linespacing=0.95)
    ax.text(50.5, 35, 'PREVENT\n5–<7.5%', fontsize=5.0, color=B3C[2], ha='center', va='center', linespacing=0.95)
    ax.axvspan(NS.loc['PREVENT >=7.5%', 'pct_tested'], NS.loc['PREVENT 5-<20%', 'pct_tested'], color=lighten(B3C[2], 0.9), lw=0, zorder=0)
    MK = {'Test all': ('s', INK), 'PREVENT 5-<20%': ('o', B3C[2]), 'PREVENT >=7.5%': ('D', BLUE), 'Age >=60 years': ('^', '#8A7FA0'),
          'PREVENT 5-<20% or age >=60': ('v', '#C07A3A'), 'Stage 2 only': ('P', '#5E5E5E')}
    for st, (mk, col) in MK.items():
        r = NS.loc[st]
        ax.errorbar(r.pct_tested, r.pct_events, yerr=[[r.pct_events - r.pct_events_lo], [r.pct_events_hi - r.pct_events]], fmt='none', ecolor=col,
                    elinewidth=0.5, zorder=3)
        ax.scatter(r.pct_tested, r.pct_events, marker=mk, s=16 if st == 'PREVENT 5-<20%' else 10, color=col, edgecolor='white', lw=0.4, zorder=4)
    r = NS.loc['PREVENT 5-<20%']
    ax.annotate(f'{r.pct_tested:.0f}% tested →\n{r.pct_events:.0f}% of flagged events', (r.pct_tested, r.pct_events), xytext=(-55, 11), textcoords='offset points',
                fontsize=5.0, color=B3C[2], fontweight='bold', ha='left', va='bottom', linespacing=0.95,
                arrowprops=dict(arrowstyle='-', lw=0.4, color=B3C[2], shrinkA=0, shrinkB=2))
    ax.set_xlim(0, 101); ax.set_ylim(0, 103); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Stage 1–2 population tested (%)', labelpad=1.2); ax.set_ylabel('Flagged 10-y CKM events captured (%)')
    clean(ax)
    # legend / table
    x0 = 49.0
    C.text(x0 + 32.5, 9.6, 'tested per\nflagged event', fontsize=5.0, color=MID, ha='center', linespacing=0.95)
    for k, (st, lab) in enumerate([('Test all', 'Test all'), ('PREVENT 5-<20%', 'PREVENT 5–<20%'), ('PREVENT >=7.5%', 'PREVENT ≥7.5%'),
                                   ('Age >=60 years', 'Age ≥60 y'), ('PREVENT 5-<20% or age >=60', 'PREVENT 5–<20% or ≥60 y'),
                                   ('Stage 2 only', 'Stage 2 only')]):
        mk, col = MK[st]; y = 15.5 + k * 4.3; r = NS.loc[st]
        kx = C.ax(x0 - 0.2, y - 1.0, 2.0, 2.0); kx.set_axis_off(); kx.set_xlim(-1, 1); kx.set_ylim(-1, 1)
        kx.scatter([0], [0], marker=mk, s=10, color=col, lw=0)
        bold = st == 'PREVENT 5-<20%'
        C.text(x0 + 2.2, y, lab, fontsize=5.0, va='center', fontweight='bold' if bold else 'normal', color=B3C[2] if bold else INK)
        C.text(x0 + 32.5, y, f'{r.tested_per_event:.1f}', fontsize=5.0, va='center', ha='center', color=B3C[2] if bold else INK, fontweight='bold' if bold else 'normal')
    for k, (st, lab) in enumerate([('Rank by PREVENT', 'ranked by PREVENT'), ('Rank by age', 'ranked by age')]):
        y = 43.0 + k * 3.2
        C.line([x0 - 0.2, x0 + 1.8], [y, y], color=RANKC[st], lw=0.9)
        C.text(x0 + 2.2, y, lab, fontsize=5.0, va='center')
    C.text(4.3, 50.5, 'Flagged event: 10-y CKM composite event in a person with ≥2 aged organs; bars, 95% CI', fontsize=5.0, color=MID)


# ============================================================================================== g  CORROBORATE
def pg(C):
    head(C, 'CORROBORATE', 'Organ-matched structural damage at the imaging visit')
    I = rd('Fig6g_imaging_associations'); DI = rd('Fig6g_damage_index')
    nn = lambda vs: int(I[I['var'].isin(vs)].n.max())
    C.text(4.3, 60.2, f'Cross-sectional, analysed n: cardiac MRI {nn(["lvef", "lvm", "laef"]):,}; aortic MRI {nn(["asc_dist", "desc_dist"]):,}; '
           f'arterial stiffness {nn(["asi"]):,}; brain MRI {nn(["tbv", "log_wmh"]):,}', fontsize=5.0, color=MID)
    ROWS = [('Heart', 'laef', 'LA ejection fraction ↓'), ('Heart', 'lvm', 'LV mass ↑'), ('Heart', 'concentric', 'LV concentricity ↑'),
            ('Heart', 'lvef', 'LV ejection fraction ↓'), ('Heart', 'gls', 'LV longitudinal strain ↓'),
            ('Artery', 'asc_dist', 'Ascending aortic distensibility ↓'), ('Artery', 'desc_dist', 'Descending aortic distensibility ↓'),
            ('Artery', 'asi', 'Arterial stiffness index ↑'),
            ('Brain', 'tbv', 'Brain volume ↓'), ('Brain', 'log_wmh', 'White matter hypointensity ↑'), ('Brain', 'fa_genu', 'FA, genu of corpus callosum ↓'),
            ('Brain', 'hippo', 'Hippocampal volume ↓'), ('Brain', 'cort_thk', 'Cortical thickness ↓')]
    COLS = ['Heart', 'Artery', 'Brain', 'Kidney', 'Liver', 'Pancreas', 'Immune', 'Lung', 'Intestine', 'Muscle', 'Organismal', 'Conventional']
    x0, y0, cw, rh = 37.0, 20.5, 4.1, 2.5
    ax = C.ax(x0, y0, cw * 12, rh * 13)
    norm = TwoSlopeNorm(vmin=-0.08, vcenter=0, vmax=0.08)
    for i, (dom, v, lab) in enumerate(ROWS):
        for j, c in enumerate(COLS):
            r = I[(I['var'] == v) & (I.clock == c)].iloc[0]
            sz = 2 + 30 * min(abs(r.beta), 0.08) / 0.08
            ax.scatter(j, i, s=sz, color=DIV(norm(r.beta)), edgecolor=INK if r.q < 0.05 else 'none', lw=0.45, zorder=3)
        C.text(x0 - 0.8, y0 + rh * (i + 0.5), lab, fontsize=5.0, ha='right', va='center')
    for dom, (i0, i1) in {'Heart': (0, 5), 'Artery': (5, 8), 'Brain': (8, 13)}.items():
        j = COLS.index(dom)
        ax.add_patch(Rectangle((j - 0.48, i0 - 0.48), 0.96, i1 - i0 - 0.04, fc='none', ec=ORGAN_COL[dom], lw=0.8, zorder=4))
        C.rect(2.6, y0 + rh * i0 + 0.3, 0.6, rh * (i1 - i0) - 0.6, fc=ORGAN_COL[dom], ec='none')
        C.text(1.8, y0 + rh * (i0 + i1) / 2, {'Heart': 'Heart', 'Artery': 'Arteries', 'Brain': 'Brain'}[dom], fontsize=5.0, rotation=90, ha='center',
               va='center', color=ORGAN_COL[dom], fontweight='bold')
        if i0 > 0:
            ax.axhline(i0 - 0.5, color=FAINT, lw=0.4)
    ax.set_xlim(-0.6, 11.6); ax.set_ylim(12.6, -0.6); ax.set_axis_off()
    for j, c in enumerate(COLS):
        C.text(x0 + cw * (j + 0.5), y0 - 0.8, c, fontsize=5.0, rotation=90, ha='center', va='bottom', color=ORGAN_COL[c])
    hcolorbar(C, DIV, norm, x0 - 26, y0 + rh * 13 + 2.2, 18, 1.2, 'β per s.d. (positive = adverse)', ticks=[-0.08, 0, 0.08],
              fmt=lambda t: um(f'{t:g}'))
    kx = C.ax(x0 + 14, y0 + rh * 13 + 1.6, 2.4, 2.4); kx.set_axis_off(); kx.set_xlim(-1, 1); kx.set_ylim(-1, 1)
    kx.scatter([0], [0], s=20, color=DIV(norm(0.06)), edgecolor=INK, lw=0.45)
    C.text(x0 + 17.0, y0 + rh * 13 + 2.8, 'FDR q <0.05;  box, organ-paired clock', fontsize=5.0, va='center', color=MID)
    # damage index
    ax2 = C.ax(97.5, y0 + 1.0, 13.0, rh * 13 - 3.0)
    rows = [('dmg_index', 'Composite'), ('dmg_heart', 'Heart'), ('dmg_artery', 'Arteries'), ('dmg_brain', 'Brain')]
    C.text(104.0, y0 - 10.0, 'Subclinical', fontsize=5.4, fontweight='bold', ha='center')
    C.text(104.0, y0 - 7.6, 'damage index', fontsize=5.4, fontweight='bold', ha='center')
    for i, (k, lab) in enumerate(rows):
        g = DI[DI['index'] == k].set_index('bcat'); y = 3 - i
        for b, dy in [(1, 0.17), (2, -0.17)]:
            r = g.loc[b]
            ci(ax2, np.array([r['diff']]), np.array([y + dy]), np.array([r.lo]), np.array([r.hi]), color=B3C[b], ms=2.2, lw=0.6)
        col = ORGAN_COL.get(lab.replace('Arteries', 'Artery'), INK)
        C.text(97.0, y0 + 1.0 + (rh * 13 - 3.0) * (3.6 - y) / 4.2 - 1.0, lab, fontsize=5.1, ha='right', va='center', color=col, fontweight='bold')
        C.text(97.0, y0 + 1.0 + (rh * 13 - 3.0) * (3.6 - y) / 4.2 + 1.3, f'P = {fmt_p(g.loc[0, "p_trend"])}', fontsize=5.0, ha='right', va='center', color=MID)
    ax2.axvline(0, color=INK, lw=0.45); ax2.set_ylim(-0.6, 3.6); ax2.set_xlim(-0.06, 0.27); ax2.set_yticks([])
    ax2.set_xticks([0, 0.2]); clean(ax2, left=False)
    ax2.set_xlabel('s.d. vs 0 aged', labelpad=1.2)
    C.text(96.0, y0 - 3.6, '●', fontsize=5.0, color=B3C[1], va='center'); C.text(98.0, y0 - 3.6, '1', fontsize=5.0, va='center')
    C.text(101.0, y0 - 3.6, '●', fontsize=5.0, color=B3C[2], va='center'); C.text(103.0, y0 - 3.6, '≥2 aged', fontsize=5.0, va='center')


# ============================================================================================== h  SYNTHESIS
def ph(C):
    head(C, 'SYNTHESIS', 'Proposed organ-age-informed C-P-R pathway')
    T = rd('Fig6b_threshold_upgrade'); OE = rd('Fig6a_oe_by_burden').set_index('bcat'); NS = rd('Fig6f_named_strategies').set_index('strategy')
    G = rd('Fig6e_risk_grid'); h_ovl = G[(G.outcome == 'OVL') & (G.bcat == 'HR_b2')].iloc[0]
    bl = T[(T.tier == '5-<7.5%') & (T.bcat == 2)].iloc[0]
    ns = NS.loc['PREVENT 5-<20%']
    cards = [
        ('CALCULATE', 'PREVENT-CVD 10-y risk', f'calibrated if 0 aged organs (O/E {OE.loc[0, "oe"]:.2f}); under-predicts if ≥2 ({OE.loc[2, "oe"]:.2f})', 'a'),
        ('PERSONALIZE', 'Borderline 5–<7.5% + ≥2 aged organs', um(f'observed {bl.obs:.1f}%: manage as ≥7.5% (ΔNB +{bl.dnb_per1000_upgraded:.0f}/1,000 upgraded)'), 'b, c'),
        ('RECLASSIFY', 'Normal routine test + aged paired organ', 'NT-proBNP or echo · cystatin C and UACR · cognitive testing', 'd'),
        ('COORDINATE', 'Any CKM stage + ≥2 aged organs', f'joint cardio–renal–metabolic follow-up (overlap HR {h_ovl.cif10:.1f})', 'e'),
        ('IMPLEMENT', 'Test organ age first in PREVENT 5–<20%', f'{ns.pct_tested:.0f}% tested captures {ns.pct_events:.0f}% of flagged events', 'f')]
    y = 9.0; hh = 7.3; gap = 1.7
    for k, (tag, l1, l2, ref) in enumerate(cards):
        C.rect(4.3, y, 61.5, hh, fc=BAND if k % 2 == 0 else 'white', ec=FAINT, lw=0.4)
        C.rect(4.3, y, 1.1, hh, fc=STEPC, ec='none')
        C.text(6.6, y + 2.3, tag, fontsize=5.0, color=STEPC, fontweight='bold', va='center')
        C.text(20.2, y + 2.3, l1, fontsize=5.4, fontweight='bold', va='center')
        C.text(64.6, y + 2.3, ref, fontsize=5.0, color=MID, ha='right', fontstyle='italic', va='center')
        C.text(6.6, y + 5.2, l2, fontsize=5.0, color=INK, va='center')
        if k < len(cards) - 1:
            C.arrow(35.0, y + hh + 0.1, 35.0, y + hh + gap - 0.1, lw=0.5, color=MID)
        y += hh + gap
    C.rect(4.3, y + 0.3, 61.5, 5.0, fc='white', ec=B3C[2], lw=0.5)
    C.text(35.0, y + 1.8, 'Proposed for prospective evaluation. Observational associations,', fontsize=5.0, color=B3C[2], ha='center', va='center')
    C.text(35.0, y + 3.8, 'not treatment effects, clinical utility or cost-effectiveness', fontsize=5.0, color=B3C[2], ha='center', va='center')


# ============================================================================================== layout
W, Hh = 180, 241
R1, R2, R3, R4 = 0, 65, 123, 179
PANELS = {
    'a': (pa, 0, R1, 61, 63),
    'b': (pb, 61, R1, 60, 63),
    'c': (pc, 121, R1, 59, 63),
    'd': (pd_, 0, R2, 180, 55),
    'e': (pe, 0, R3, 92, 53),
    'f': (pf, 92, R3, 88, 53),
    'g': (pg, 0, R4, 113, 62),
    'h': (ph, 113, R4, 67, 62),
}
if __name__ == '__main__':
    only = sys.argv[1:]
    if only:
        for lab in only:
            fn, x, y, w, h = PANELS[lab]
            single(fn, w, h, f'Fig6{lab}', os.path.join(ROOT, 'panels'), lab=lab)
    else:
        build(PANELS, W, Hh, 'Fig6', ROOT)
    print('done')
