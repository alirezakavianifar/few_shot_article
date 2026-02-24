import json
nb = json.loads(open('dekae_colab.ipynb', encoding='utf-8').read())
for c in nb['cells']:
    if c.get('id') == '1f9b5c4b':
        s = ''.join(c.get('source', ''))
        open('cell36.txt', 'w', encoding='utf-8').write(s)
        print('Written')
        break
