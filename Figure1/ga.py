"""Helpers for graphical-abstract elements (Figure 1, v2).

Icons: Health Icons (https://healthicons.org), CC0 1.0 public domain, filled set (48 x 48 viewBox).
They are converted to vector paths so that the PDF output remains fully vector.
All geometry is in millimetres from the top-left corner of a panel (style4.Canvas convention).
"""
import os
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from svgelements import SVG, Shape, Path as SPath
from svgpath2mpl import parse_path
from style4 import INK, MID, LIGHT, FAINT, lighten

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
_CACHE = {}


def load_icon(name):
    """Return list of (vertices, codes) in icon units (0-48, y downwards)."""
    if name in _CACHE:
        return _CACHE[name]
    svg = SVG.parse(os.path.join(ICON_DIR, name + '.svg'))
    vb = svg.viewbox
    scale = 48.0 / (vb.width if vb is not None else 48.0)
    parts = []
    for el in svg.elements():
        if isinstance(el, Shape) and not isinstance(el, SVG):
            try:
                p = SPath(el)
                p.reify()
                d = p.d()
            except Exception:
                continue
            if not d:
                continue
            mp = parse_path(d)
            parts.append((mp.vertices * scale, mp.codes))
    # merge to a single compound path so that holes (even-odd) render correctly
    V = np.concatenate([v for v, c in parts])
    Cc = np.concatenate([c for v, c in parts])
    _CACHE[name] = (V, Cc)
    return _CACHE[name]


def icon(C, name, x, y, size, color=INK, alpha=1.0, zorder=5):
    """Draw icon with its top-left corner at (x, y) mm and width/height `size` mm."""
    V, codes = load_icon(name)
    xs = np.array([C._fx(x + vx / 48.0 * size) for vx in V[:, 0]])
    ys = np.array([C._fy(y + vy / 48.0 * size) for vy in V[:, 1]])
    p = Path(np.column_stack([xs, ys]), codes)
    patch = PathPatch(p, transform=C.fig.transFigure, fc=color, ec='none', alpha=alpha, zorder=zorder)
    C.fig.add_artist(patch)
    return patch


def rrect(C, x, y, w, h, r=1.2, fc='white', ec=FAINT, lw=0.5, zorder=1, ls='-', alpha=1.0):
    """Rounded rectangle drawn in mm space (undistorted corners)."""
    r = min(r, w / 2, h / 2)
    k = 0.5523
    pts = [(x + r, y), (x + w - r, y),
           (x + w - r + k * r, y), (x + w, y + r - k * r), (x + w, y + r),
           (x + w, y + h - r),
           (x + w, y + h - r + k * r), (x + w - r + k * r, y + h), (x + w - r, y + h),
           (x + r, y + h),
           (x + r - k * r, y + h), (x, y + h - r + k * r), (x, y + h - r),
           (x, y + r),
           (x, y + r - k * r), (x + r - k * r, y), (x + r, y), (x + r, y)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    P = Path([(C._fx(a), C._fy(b)) for a, b in pts], codes)
    patch = PathPatch(P, transform=C.fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=zorder, ls=ls, alpha=alpha)
    C.fig.add_artist(patch)
    return patch


def circle(C, x, y, r, fc='white', ec=INK, lw=0.5, zorder=3):
    t = np.linspace(0, 2 * np.pi, 80)
    P = Path([(C._fx(x + r * np.cos(a)), C._fy(y + r * np.sin(a))) for a in t])
    patch = PathPatch(P, transform=C.fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=zorder)
    C.fig.add_artist(patch)
    return patch


def chevron(C, x, y, h, w=1.6, color=LIGHT, lw=0.8):
    """Right-pointing chevron with its tip at (x, y)."""
    C.line([x - w, x, x - w], [y - h / 2, y, y + h / 2], color=color, lw=lw, solid_capstyle='round',
           solid_joinstyle='round')


def kicker(C, x, y, s, color=MID):
    """Small category label above a panel title (sentence case, per Nature Communications figure style)."""
    C.text(x, y, s[:1].upper() + s[1:], fontsize=5.4, color=color, fontweight="bold", va="top", ha="left")


def title(C, x, y, s, size=6.6):
    C.text(x, y, s, fontsize=size, fontweight='bold', va='top', ha='left')
