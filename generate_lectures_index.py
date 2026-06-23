# -*- coding: utf-8 -*-
"""Generate public/lectures/index.html — a browsable manifest of the AI503 course materials."""
import os, html
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
LEC = os.path.join(HERE, 'lectures')

WEEK_TITLES = {
    'W01-W02': 'Weeks 1–2 · Logistic Regression',
    'W03': 'Week 3 · KNN · Decision Trees · SVM · Naive Bayes',
    'W04': 'Week 4 · Clustering (K-Means)',
    'W05-W06': 'Weeks 5–6 · Deep Learning',
    'W07': 'Week 7 · LSTM · Model Comparison',
    'W08-W09': 'Weeks 8–9 · Ensemble Learning + CNN Extra Assignment',
}
ICON = {'.pdf': '📑', '.ipynb': '📓', '.html': '🌐', '.csv': '📊',
        '.py': '🐍', '.md': '📝', '.odt': '📄', '.zip': '🗜', '.png': '🖼', '.jpg': '🖼'}

def icon(name):
    return ICON.get(os.path.splitext(name)[1].lower(), '📎')

def li(relpath, label):
    href = '/'.join(quote(p) for p in relpath.split('/'))
    return f'      <li>{icon(label)} <a href="{href}">{html.escape(label)}</a></li>'

rows = []
for wk in sorted(os.listdir(LEC)):
    wd = os.path.join(LEC, wk)
    if not os.path.isdir(wd):
        continue
    rows.append(f'  <h2>{html.escape(WEEK_TITLES.get(wk, wk))}</h2>')
    # direct files in the week folder
    direct = sorted(f for f in os.listdir(wd) if os.path.isfile(os.path.join(wd, f)))
    if direct:
        rows.append('    <ul>')
        for f in direct:
            rows.append(li(f'{wk}/{f}', f))
        rows.append('    </ul>')
    # the CNN Extra Assignment lives under W08-W09/Assignment_Extra — surface its key deliverables
    ax = os.path.join(wd, 'Assignment_Extra')
    if os.path.isdir(ax):
        rows.append('    <div class="extra"><strong>📦 CNN Image Classification — Extra Assignment</strong>'
                    '<ul>')
        for rel, lbl in [
            ('Assignment_Extra/report/final/A2_AI503_CNN_Report.pdf', 'Written report (PDF, 43 pp)'),
            ('Assignment_Extra/notebook/CNN_Image_Classification_CIFAR10.html', 'Notebook — read-only HTML'),
            ('Assignment_Extra/notebook/CNN_Image_Classification_CIFAR10.ipynb', 'Notebook — .ipynb (run on GPU)'),
            ('Assignment_Extra/records/cnn_leaderboard_results.csv', 'Leaderboard results (CSV)'),
            ('Assignment_Extra/README.md', 'Assignment README'),
        ]:
            if os.path.exists(os.path.join(wd, rel)):
                rows.append(li(f'{wk}/{rel}', lbl))
        rows.append('    </ul></div>')

doc = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI503 — Course Lectures &amp; Labs</title>
<style>
 body {{ font-family:-apple-system,Segoe UI,system-ui,sans-serif; margin:0 auto; max-width:880px;
        padding:32px; background:#f7f8fa; color:#1f2329; }}
 h1 {{ font-size:24px; margin:0 0 4px; }} .sub {{ color:#555; font-size:14px; margin-bottom:20px; }}
 h2 {{ font-size:17px; color:#4472C4; margin:22px 0 6px; border-bottom:1px solid #e1e4e8; padding-bottom:4px; }}
 ul {{ margin:6px 0 6px 4px; padding-left:20px; }} li {{ margin:3px 0; font-size:14px; }}
 a {{ color:#4472C4; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
 .extra {{ background:#fff; border:1px solid #e1e4e8; border-left:4px solid #4472C4;
          border-radius:8px; padding:12px 16px; margin:8px 0; }}
 .back {{ font-size:14px; }} .footer {{ margin-top:28px; font-size:12px; color:#888; text-align:center; }}
</style></head><body>
<p class="back"><a href="../index.html">&larr; AI503 home</a></p>
<h1>AI503 — Machine Learning · Course Lectures &amp; Labs</h1>
<p class="sub">Lecture slides, lab notebooks, datasets, and the CNN Extra Assignment · BUiD Spring 2026 · Mohamed Elrashid</p>
{chr(10).join(rows)}
<div class="footer">Lecture slides © their authors, shared for study. Source:
 <a href="https://github.com/Elrashid/AI503">github.com/Elrashid/AI503</a></div>
</body></html>
'''
open(os.path.join(LEC, 'index.html'), 'w', encoding='utf-8').write(doc)
print('wrote lectures/index.html (%d weeks)' % sum(1 for w in WEEK_TITLES if os.path.isdir(os.path.join(LEC, w))))
