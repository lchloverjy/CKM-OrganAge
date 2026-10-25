"""Check that the baseline protein file and the organ age gap file contain the same participants.
Writes results/v5_fig1v2/F1_protein_agegap_match.json (run from the v5 run/ directory)."""
import json
import pandas as pd, pyreadr
P = '/home/ubuntu/upload/ukb_protein_wide_ins0_final.rds'
G = '/home/ubuntu/upload/age_gap.xlsx'
prot = list(pyreadr.read_r(P).values())[0]
gap = pd.read_excel(G)
ep, eg = set(prot['eid'].astype(int)), set(gap['eid'].astype(int))
gap_cols = [c for c in gap.columns if c != 'eid']
res = dict(protein_file_participants=len(prot), protein_file_unique_eid=len(ep), n_proteins=prot.shape[1] - 1,
           age_gap_participants=len(gap), age_gap_unique_eid=len(eg), n_age_gap_columns=len(gap_cols),
           age_gap_complete_rows=int(gap[gap_cols].notna().all(axis=1).sum()),
           overlap=len(ep & eg), protein_only=len(ep - eg), age_gap_only=len(eg - ep))
json.dump(res, open('results/v5_fig1v2/F1_protein_agegap_match.json', 'w'), indent=1)
print(res)
assert res['overlap'] == res['protein_file_participants'] == res['age_gap_participants']
