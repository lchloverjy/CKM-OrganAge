"""Build the v5 manuscript with Figure 1 v2 and the accompanying text edits.
Runs the v5 builder (run/code/build_v5_docx.py) on ../integrated/ms_v5_fig1v2_markup.txt with Fig. 1 replaced by ../Fig1.png;
Figures 2-6 and Table 1 are the v5 versions. The v5 files are not modified."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.abspath(os.path.join(HERE, '..', 'integrated'))
src = open('/home/ubuntu/work/v5/run/code/build_v5_docx.py', encoding='utf-8').read()
src = src.replace("MARK = '/home/ubuntu/work/v5/ms/ms_v5_markup.txt'", f"MARK = '{INT}/ms_v5_fig1v2_markup.txt'")
src = src.replace("FIGS = {i: RUN + f'v4/Figure{i}/Fig{i}.png' for i in range(1, 7)}",
                  "FIGS = {i: RUN + f'v4/Figure{i}/Fig{i}.png' for i in range(1, 7)}\n"
                  f"FIGS[1] = '{os.path.abspath(os.path.join(HERE, '..', 'Fig1.png'))}'")
assert 'ms_v5_fig1v2_markup' in src and "FIGS[1] = " in src
tmp = '/tmp/_build_ms_fig1v2.py'
open(tmp, 'w', encoding='utf-8').write(src)
sys.path.insert(0, '/home/ubuntu/work/v5/run/code')
env = dict(os.environ, PYTHONPATH='/home/ubuntu/work/v5/run/code')
subprocess.run(['python3', tmp, f'{INT}/Manuscript_CKM_organ_age_v5_fig1v2.docx'], check=True, env=env)
