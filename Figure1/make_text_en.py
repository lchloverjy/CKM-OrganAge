"""Write text/02_Fig1_Results_Legend_Methods_EN.md from the integrated markup and the SI build inputs,
so that the English text is identical to what is in the integrated manuscript and Supplementary Information."""
import os, json, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.join(HERE, '..', 'integrated')
mk = open(os.path.join(INT, 'ms_v5_fig1v2_markup.txt'), encoding='utf-8').read().split('\n')
old = open('/home/ubuntu/work/v5/ms/ms_v5_markup.txt', encoding='utf-8').read().split('\n')
B = os.path.join(INT, '_si_build', 'v4')
TB = json.load(open(B + '/Supplementary/Supplementary_Tables/tables.json'))
NOTES = json.load(open(B + '/manuscript/supp_notes.json'))


def line(start):
    x = [l for l in mk if l.startswith(start)]
    assert x, start
    return x[0]


def sentence_with(key):
    for l in mk:
        if key in l:
            for s in re.split(r'(?<=\.)\s+(?=[A-Z])', l):
                if key in s:
                    return s
    raise KeyError(key)


changed = sorted(set(l for l in mk if not l.startswith('@@')) - set(l for l in old if not l.startswith('@@')))
out = []
w = out.append
w('# Figure 1 (round 4, concise): Results, figure legend, Methods and Supplementary text (English)\n')
w('> Generated from `integrated/ms_v5_fig1v2_markup.txt` and the Supplementary Information build inputs '
  '(`code/make_text_en.py`); the text below is identical to that in the integrated manuscript and Supplementary Information. '
  f'{len(changed)} manuscript paragraphs differ from v5.\n')
w('## 1. Results — first paragraph of "Study population and outcome system" (replaces the v5 paragraph)\n')
w(line('The analytic cohort included 40,674') + '\n')
w('## 2. Figure 1 legend (replaces both copies in v5)\n')
w(line('**Fig. 1 | Study overview') + '\n')
w(line('**a**, Follow-up via hospital and death records') + '\n')
w('## 3. Methods\n')
w('**Study population (revised sentences).** ' + sentence_with('leaving 40,674 participants in stages 0–3 (Fig. 1a') + ' '
  + sentence_with('Characteristics of included and excluded participants') + ' '
  + sentence_with('the sample of every analysis is listed in Supplementary Table 1b') + '\n')
w('**CKM staging.** "(Supplementary Table 1)" → "(Supplementary Table 1c)".\n')
w('**Plasma proteomics and organ age gaps (added sentence).** ' + sentence_with('The structure of the 12 age gaps') + '\n')
w('**Cross-reference corrections carried over from v5.** "Reporting follows the STROBE guideline (Supplementary Table 21)" → '
  '"(Supplementary Table 23)"; "three (cataract, glaucoma and obesity-related cancers) to the organismal clock (Supplementary '
  'Table 2)" → "(Supplementary Table 4a)".\n')
w('## 4. Supplementary Note 1 (new first paragraph; title revised)\n')
w(f'**Supplementary Note 1. {NOTES[0][0]}**\n')
w(NOTES[0][1][0] + '\n')
w('## 5. Supplementary Table 1\n')
t = TB[0]
w(f"**Supplementary Table 1. {t['title']}**\n")
w('Parts: ' + '; '.join(f"{p['label']} ({p['rows']} rows)" for p in t['parts']) + '. Full tables: '
  '`supplementary/Supplementary_Table_1.pdf` and `.xlsx`.\n')
w('**Table note.** ' + t['note'] + '\n')
a = pd.read_csv(B + '/Supplementary/Supplementary_Tables/display/ST1a.csv', dtype=str, keep_default_na=False)
w('**Supplementary Table 1a (participant selection).**\n')
w('| ' + ' | '.join(a.columns) + ' |')
w('|' + '---|' * a.shape[1])
for _, r in a.iterrows():
    w('| ' + ' | '.join(r.values) + ' |')
w('')
w('## 6. STROBE checklist (Supplementary Table 23), revised locations\n')
st = pd.read_csv(B + '/Supplementary/Supplementary_Tables/display/ST23.csv', dtype=str, keep_default_na=False)
st = st[st.Item.isin(['6', '10', '13', '14'])]
w('| Item | Recommendation | Location in manuscript |')
w('|---|---|---|')
for _, r in st.iterrows():
    w(f"| {r.Item} | {r.Recommendation} | {r['Location in manuscript']} |")
w('')
w('## 7. Items for the authors\n')
w('1. Reference numbers for Nyholt (Am. J. Hum. Genet. 2004) and Li and Ji (Heredity 2005) in Methods (placeholder '
  '"[reference numbers to be added]").\n'
  '2. Optional sensitivity analysis restricted to the randomly selected UKB-PPP participants.\n')
open(os.path.join(HERE, '..', 'text', '02_Fig1_Results_Legend_Methods_EN.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('written; changed paragraphs:', len(changed))
