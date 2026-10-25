"""Regenerates results/L2_smd_highburden.csv (script missing from the original package).
Definition verified to reproduce the original file exactly when run with the provider stage flags:
stages 1-2; high burden = >=2 aged target organs; L1.load() median-imputed values; SMD with pooled (ddof=1) variance."""
import sys; sys.path.insert(0, 'analysis'); sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from importlib import import_module
L1 = import_module('02_layer1'); L2 = import_module('03_layer2')
b = L1.load(); b = b[b.stage.isin([0, 1, 2, 3])]
F = L2.extreme_flags(b); bt = F[L2.TARGET].sum(1).values
V = {'PREVENT 10-y CVD risk': 'prevent10', 'Age': 'age', 'eGFR': 'egfr', 'HbA1c': 'hba1c', 'BMI': 'bmi', 'Waist': 'waist',
     'SBP': 'sbp', 'TG/HDL': 'tg_hdl', 'CRP': 'crp', 'NLR': 'nlr'}
m = b.stage.isin([1, 2]).values; hi = (bt >= 2) & m; lo = (bt < 2) & m
rows = []
for lab, v in V.items():
    x = b[v].values; h = x[hi]; l = x[lo]; h = h[~np.isnan(h)]; l = l[~np.isnan(l)]
    rows.append(dict(trait=lab, smd=(h.mean() - l.mean()) / np.sqrt((h.var(ddof=1) + l.var(ddof=1)) / 2), med_high=np.median(h), med_low=np.median(l)))
pd.DataFrame(rows).to_csv('./results/L2_smd_highburden.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string(), hi.sum(), lo.sum())
