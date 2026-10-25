"""Shared plotting conventions for the CKM organ-age figures (version 4).

All geometry is in millimetres measured from the top-left corner of a panel.
Each panel is drawn by a function ``f(C)`` that receives a Canvas.  The same
function is used to export the panel alone and to place it in the composite
figure, so the two are identical.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

MM = 1 / 25.4

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6,
    'axes.titlesize': 6.5,
    'axes.labelsize': 6.5,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#222222',
    'axes.labelcolor': '#111111',
    'axes.labelpad': 2.0,
    'xtick.labelsize': 5.8,
    'ytick.labelsize': 5.8,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'xtick.minor.size': 1.2,
    'ytick.minor.size': 1.2,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'legend.fontsize': 5.8,
    'legend.frameon': False,
    'legend.handlelength': 1.2,
    'legend.handletextpad': 0.4,
    'legend.borderaxespad': 0.2,
    'lines.linewidth': 0.9,
    'patch.linewidth': 0.5,
    'pdf.fonttype': 42,
    'axes.unicode_minus': True,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.dpi': 600,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.default': 'regular',
})

INK = '#1A1A1A'
MID = '#5E5E5E'
LIGHT = '#9A9A9A'
FAINT = '#D9D9D9'
BAND = '#F2F2F2'

STAGE_COL = ['#9AA7B4', '#E3B23C', '#D9732B', '#A4161A']
STAGE_LAB = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3']

ORGAN_COL = {
    'Heart': '#B23A48', 'Artery': '#E07A5F', 'Kidney': '#3D5A80', 'Liver': '#8D6346',
    'Pancreas': '#D4A017', 'Brain': '#6D597A', 'Immune': '#2A9D8F', 'Lung': '#5FA8D3',
    'Intestine': '#C77DBA', 'Muscle': '#6B8F71', 'Organismal': '#343A40',
    'Conventional': '#9E9E9E',
}
TARGET = ['Kidney', 'Artery', 'Heart', 'Liver', 'Pancreas', 'Brain']
CLOCKS = ['Kidney', 'Immune', 'Muscle', 'Artery', 'Liver', 'Intestine', 'Heart',
          'Pancreas', 'Organismal', 'Lung', 'Brain', 'Conventional']

ACCENT = '#A4161A'
BLUE = '#2F5D8C'

DIV = LinearSegmentedColormap.from_list(
    'div', ['#1F4E89', '#4F81B8', '#A9C6E3', '#F4F4F4', '#F2B7A5', '#D6604D', '#8E1B1B'])
SEQ = LinearSegmentedColormap.from_list(
    'seq', ['#FBF6F0', '#F6D7C3', '#EDA184', '#D9634B', '#A62B2B', '#5E0E14'])
SEQ_BLUE = LinearSegmentedColormap.from_list(
    'seqb', ['#F5F8FB', '#C9DAEA', '#8FB3D5', '#4F81B8', '#1F4E89', '#0E2A4E'])


def lighten(c, f=0.5):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


class Canvas:
    def __init__(self, fig, ox, oy, FW, FH):
        self.fig, self.ox, self.oy, self.FW, self.FH = fig, ox, oy, FW, FH

    def _fx(self, x):
        return (self.ox + x) / self.FW

    def _fy(self, y):
        return 1 - (self.oy + y) / self.FH

    def ax(self, x, y, w, h, **kw):
        a = self.fig.add_axes([self._fx(x), self._fy(y + h), w / self.FW, h / self.FH], **kw)
        a.tick_params(which='both', direction='out')
        return a

    def text(self, x, y, s, **kw):
        kw.setdefault('fontsize', 6)
        kw.setdefault('color', INK)
        kw.setdefault('va', 'top')
        return self.fig.text(self._fx(x), self._fy(y), s, **kw)

    def line(self, xs, ys, **kw):
        kw.setdefault('lw', 0.5)
        kw.setdefault('color', INK)
        l = Line2D([self._fx(x) for x in xs], [self._fy(y) for y in ys],
                   transform=self.fig.transFigure, **kw)
        self.fig.add_artist(l)
        return l

    def rect(self, x, y, w, h, **kw):
        kw.setdefault('lw', 0.5)
        kw.setdefault('fc', 'none')
        kw.setdefault('ec', INK)
        r = Rectangle((self._fx(x), self._fy(y + h)), w / self.FW, h / self.FH,
                      transform=self.fig.transFigure, **kw)
        self.fig.add_artist(r)
        return r

    def arrow(self, x0, y0, x1, y1, **kw):
        kw.setdefault('lw', 0.6)
        kw.setdefault('color', INK)
        kw.setdefault('arrowstyle', '-|>,head_length=2.2,head_width=1.1')
        kw.setdefault('shrinkA', 0)
        kw.setdefault('shrinkB', 0)
        a = FancyArrowPatch((self._fx(x0), self._fy(y0)), (self._fx(x1), self._fy(y1)),
                            transform=self.fig.transFigure, mutation_scale=1, **kw)
        self.fig.add_artist(a)
        return a


def letter(C, s, x=0.0, y=0.0):
    C.text(x, y, s, fontsize=8, fontweight='bold', va='top', ha='left')


def clean(ax, left=True, bottom=True, grid=None):
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    if not left:
        ax.tick_params(axis='y', length=0)
    if not bottom:
        ax.tick_params(axis='x', length=0)
    if grid == 'x':
        ax.grid(axis='x', color=FAINT, lw=0.35, zorder=0)
        ax.set_axisbelow(True)
    elif grid == 'y':
        ax.grid(axis='y', color=FAINT, lw=0.35, zorder=0)
        ax.set_axisbelow(True)
    return ax


def ci(ax, x, y, lo, hi, color=INK, horizontal=True, ms=2.6, lw=0.7, filled=True,
       marker='o', zorder=3, cap=0):
    if horizontal:
        ax.errorbar(x, y, xerr=[np.atleast_1d(x - lo), np.atleast_1d(hi - x)], fmt='none',
                    ecolor=color, elinewidth=lw, capsize=cap, zorder=zorder)
    else:
        ax.errorbar(x, y, yerr=[np.atleast_1d(y - lo), np.atleast_1d(hi - y)], fmt='none',
                    ecolor=color, elinewidth=lw, capsize=cap, zorder=zorder)
    ax.scatter(x, y, s=ms ** 2, marker=marker, facecolor=color if filled else 'white',
               edgecolor=color, linewidth=0.6, zorder=zorder + 1)


def hcolorbar(C, cmap, norm, x, y, w, h, label, ticks=None, fmt=None, labelpad=1.0):
    cax = C.ax(x, y, w, h)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.outline.set_linewidth(0.4)
    cax.tick_params(length=1.5, width=0.4, labelsize=5.5, pad=1)
    if ticks is not None:
        cb.set_ticks(ticks)
    if fmt is not None:
        cb.set_ticklabels([fmt(t) for t in ticks])
    cax.set_title(label, fontsize=5.8, pad=labelpad)
    return cb


def _save(fig, path_noext, tif=False):
    fig.savefig(path_noext + '.pdf')
    fig.savefig(path_noext + '.png', dpi=600)
    if tif:
        from PIL import Image
        im = Image.open(path_noext + '.png').convert('RGB')
        im.save(path_noext + '.tif', compression='tiff_lzw', dpi=(600, 600))


def single(fn, w, h, name, outdir, lab=None):
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(w * MM, h * MM))
    C = Canvas(fig, 0, 0, w, h)
    fn(C)
    if lab:
        letter(C, lab)
    _save(fig, os.path.join(outdir, name))
    plt.close(fig)


def compose(items, W, H, name, outdir):
    """items: list of (fn, x, y, letter)."""
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(W * MM, H * MM))
    for fn, x, y, lab in items:
        C = Canvas(fig, x, y, W, H)
        fn(C)
        if lab:
            letter(C, lab)
    _save(fig, os.path.join(outdir, name), tif=True)
    plt.close(fig)


def build(panels, W, H, name, root):
    """panels: dict letter -> (fn, x, y, w, h)."""
    for lab, (fn, x, y, w, h) in panels.items():
        single(fn, w, h, f'{name}{lab}', os.path.join(root, 'panels'), lab=lab)
    compose([(fn, x, y, lab) for lab, (fn, x, y, w, h) in panels.items()], W, H, name, root)


def fmt_p(p):
    if p < 0.001:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        return f'{m:.1f}×10$^{{{e}}}$'
    return f'{p:.2g}' if p < 0.01 else f'{p:.2f}'


def um(s):
    """Replace ASCII hyphen-minus used as a numeric sign by the typographic minus sign."""
    import re
    return re.sub(r'(?<![A-Za-z])-(?=\d)', '\u2212', s)
