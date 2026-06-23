# -*- coding: utf-8 -*-
"""Add reference IDs [R1..R16] + verified DOI links to the reference list, and tag every
inline (Author, Year) citation with its [Rn]. Idempotent. DOIs verified via Crossref/DataCite."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'REPORT.md')

# surname-at-start  ->  (id, link, kind)   (kind: doi | url | none)
REFMAP = {
    'Breiman':    ('R1',  'https://doi.org/10.1007/BF00058655',            'doi'),
    'Deng':       ('R2',  'https://doi.org/10.1109/CVPR.2009.5206848',     'doi'),
    'He':         ('R3',  'https://doi.org/10.1109/CVPR.2016.90',          'doi'),
    'Howard':     ('R4',  'https://doi.org/10.48550/arXiv.1704.04861',     'doi'),
    'Ioffe':      ('R5',  'https://doi.org/10.48550/arXiv.1502.03167',     'doi'),
    'Kingma':     ('R6',  'https://doi.org/10.48550/arXiv.1412.6980',      'doi'),
    'Krizhevsky': ('R7',  'https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf', 'url'),
    'LeCun':      ('R8',  'https://doi.org/10.1109/5.726791',              'doi'),
    'Liu':        ('R9',  'https://doi.org/10.1109/CVPR52688.2022.01167',  'doi'),
    'Sandler':    ('R10', 'https://doi.org/10.1109/CVPR.2018.00474',       'doi'),
    'Selvaraju':  ('R11', 'https://doi.org/10.1109/ICCV.2017.74',          'doi'),
    'Simonyan':   ('R12', 'https://doi.org/10.48550/arXiv.1409.1556',      'doi'),
    'Srivastava': ('R13', 'https://jmlr.org/papers/v15/srivastava14a.html', 'url'),
    'Tan':        ('R14', 'https://doi.org/10.48550/arXiv.2104.00298',     'doi'),
    'Wolpert':    ('R15', 'https://doi.org/10.1016/S0893-6080(05)80023-1', 'doi'),
    'Course':     ('R16', None,                                            'none'),
}

# inline (Author, Year) token  ->  id
INLINE = [
    ('Breiman, 1996', 'R1'), ('Deng et al., 2009', 'R2'), ('He et al., 2016', 'R3'),
    ('Howard et al., 2017', 'R4'), ('Ioffe and Szegedy, 2015', 'R5'),
    ('Kingma and Ba, 2014', 'R6'), ('Krizhevsky, 2009', 'R7'),
    ('LeCun et al., 1998', 'R8'), ('Liu et al., 2022', 'R9'),
    ('Sandler et al., 2018', 'R10'), ('Selvaraju et al., 2017', 'R11'),
    ('Simonyan and Zisserman, 2014', 'R12'), ('Srivastava et al., 2014', 'R13'),
    ('Tan and Le, 2021', 'R14'), ('Wolpert, 1992', 'R15'),
]

text = open(REPORT, encoding='utf-8').read()

# --- split out the References block (refs end at the first '---' or comment after the heading) ---
ref_start = text.index('## References')
m_end = re.search(r'\n---\n|\n<!--', text[ref_start:])
ref_end = ref_start + m_end.start() if m_end else len(text)
before, refs_block, after = text[:ref_start], text[ref_start:ref_end], text[ref_end:]

# 1) inline citation IDs (only in the body before the references; idempotent via lookahead)
n_inline = 0
for token, rid in INLINE:
    new, k = re.subn(re.escape(token) + r'(?!\s*\[R)', token + ' [%s]' % rid, before)
    before = new; n_inline += k
# course-notes inline citations carry a page number, e.g. "Course Notes, W05-W06, p.18"
before, k = re.subn(r'(Course Notes, W05–W06, p\.\d+)(?!\s*\[R)', r'\1 [R16]', before)
n_inline += k

# 2) reference IDs + DOI links
out, n_ref = [], 0
for ln in refs_block.split('\n'):
    s = ln.strip()
    m = re.match(r'\*?([A-Z][A-Za-z]+)', s)
    if s and not s.startswith(('#', '<!--', '---', '**[')) and m and m.group(1) in REFMAP:
        rid, link, kind = REFMAP[m.group(1)]
        new = '**[%s]** %s' % (rid, ln)
        if kind == 'doi':
            new += ' ' + link
        elif kind == 'url':
            new += ' Available at: ' + link
        else:
            new += ' (internal teaching material; no DOI)'
        out.append(new); n_ref += 1
    else:
        out.append(ln)
refs_block = '\n'.join(out)

open(REPORT, 'w', encoding='utf-8').write(before + refs_block + after)
print('inline citations tagged: %d  |  references given ID + DOI: %d' % (n_inline, n_ref))
