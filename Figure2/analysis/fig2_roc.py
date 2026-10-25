"""Cross-validated ROC curves of the five nested models identifying high multi-organ ageing burden (>=2 aged target
organs) in CKM stages 1-2 (Fig. 2g). Models, folds and seeds are identical to fig2_text_checks.py (5-fold stratified CV,
5 repeats, out-of-fold predictions averaged across repeats), so the AUCs reproduce Supplementary Table 9c.
Output: results/v4/F2_roc.csv (ROC curve interpolated on a common false-positive-rate grid)."""
import sys, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, './code')
from importlib import import_module
from stats_utils import cv_auc
warnings.filterwarnings('ignore')
L1 = import_module('02_layer1')

a = L1.load().reset_index(drop=True)
bu = pd.read_parquet('./data/burden.parquet')
a = a.merge(bu, on='eid', how='left')
b = a[a.stage.isin([1, 2])].copy().reset_index(drop=True)
y = (b.burden_t >= 2).astype(int).values
b['log_prevent'] = np.log(b.prevent10)
base = ['age', 'age2', 'male']
clin_no_egfr = ['bmi', 'waist', 'hba1c', 'sbp', 'dbp', 'log_tg_hdl', 'ldl', 'hdl', 'log_crp', 'log_nlr', 'htn_med',
                'statin', 'diabetes', 'smoke1', 'smoke2', 'townsend', 'degree']
feats = {'Age, sex': base,
         '+ CKM stage': base + ['st2'],
         '+ PREVENT risk': base + ['st2', 'log_prevent'],
         '+ clinical panel without eGFR': base + ['st2', 'log_prevent'] + clin_no_egfr,
         '+ eGFR (full clinical panel)': base + ['st2', 'log_prevent'] + clin_no_egfr[:8] + ['egfr'] + clin_no_egfr[8:]}
grid = np.linspace(0, 1, 201)
rows = []
for k, c in feats.items():
    p, _ = cv_auc(b[c].values, y, n_repeats=5)
    fpr, tpr, _ = roc_curve(y, p)
    t = np.interp(grid, fpr, tpr); t[0] = 0.0
    auc = roc_auc_score(y, p)
    rows += [dict(model=k, fpr=f, tpr=v, auc=auc) for f, v in zip(grid, t)]
    print(f'{k:35s} AUC {auc:.4f}')
R = pd.DataFrame(rows)
R.to_csv('./results/v4/F2_roc.csv', index=False)
print('n =', len(y), 'high burden =', int(y.sum()))
