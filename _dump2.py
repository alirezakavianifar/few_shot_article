import json
nb = json.loads(open('dekae_colab.ipynb', encoding='utf-8').read())
for cid, fname in [('b9806c81', 'cell_model.txt'), ('88ed9b71', 'cell_build.txt'), ('444b4113', 'cell_eval.txt')]:
    for c in nb['cells']:
        if c.get('id') == cid:
            s = ''.join(c.get('source', ''))
            open(fname, 'w', encoding='utf-8').write(s)
            print(f'Written {fname}')
            break
