cd /home/ubuntu/work/v5/figure1/integrated
timeout 900 soffice --headless --convert-to pdf --outdir . Supplementary_Information_v5_fig1v2.docx > /dev/null 2>&1
timeout 600 soffice --headless --convert-to pdf --outdir . Manuscript_CKM_organ_age_v5_fig1v2.docx > /dev/null 2>&1
python3 - <<'PY'
import subprocess
f='Supplementary_Information_v5_fig1v2.pdf'
n=int([l for l in subprocess.run(['pdfinfo',f],capture_output=True,text=True).stdout.split('\n') if l.startswith('Pages')][0].split()[-1])
pg=[]
for p in range(1,n+1):
    t=subprocess.run(['pdftotext','-f',str(p),'-l',str(p),'-layout',f,'-'],capture_output=True,text=True).stdout
    if 'Contents' in t: continue
    if 'Study population: participant selection' in t: pg.append(p)
    if pg and 'Supplementary Table 2. Definitions' in t: end=p-1; break
s=pg[0]
subprocess.run(['qpdf','--empty','--remove-unreferenced-resources=yes','--pages',f,f'{s}-{end}','--','--object-streams=generate','../supplementary/Supplementary_Table_1.pdf'],check=True)
print('ST1 pages',s,end)
PY
echo done
