import json
nb = json.loads(open('dekae_colab.ipynb', encoding='utf-8').read())
cells = nb['cells']

def cell_src(cid):
    for c in cells:
        if c.get('id') == cid:
            return ''.join(c.get('source', []))
    return ''

checks = [
    ('EdgeIncidenceModule', 'c526db16', 'edge_proj_type'),
    ('DEKAEModel', 'b9806c81', 'edge_proj_type'),
    ('build_model', '88ed9b71', 'edge_proj_type'),
    ('ABLATION_CONFIGS', '1f9b5c4b', 'C1_no_edge_feat'),
    ('ABLATION_CONFIGS', '1f9b5c4b', 'C2_linear_proj'),
    ('ABLATION_CONFIGS', '1f9b5c4b', 'C3_mlp_proj'),
    ('Section 15b', 'a48be1a2', 'RUN_GROUP_C'),
    ('Section 15b', 'a48be1a2', 'group_c_results'),
]

all_ok = True
for label, cid, kw in checks:
    found = kw in cell_src(cid)
    status = 'OK' if found else 'MISSING'
    print('%s  [%s]  %s: %r' % (status, cid, label, kw))
    if not found:
        all_ok = False

# Check new cells exist
for new_id in ('grpc0001', 'grpc0002'):
    found = any(c.get('id') == new_id for c in cells)
    status = 'OK' if found else 'MISSING'
    print('%s  [%s]  new cell exists' % (status, new_id))
    if not found:
        all_ok = False

print()
print('All checks passed!' if all_ok else 'Some checks FAILED.')
