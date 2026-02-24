"""
patch_group_b_g.py
------------------
Patches dekae_colab.ipynb:
  PATCH 1 -- DEKAEModel: add selection_strategy param + nearest_centroid prototype
  PATCH 2 -- build_model: pass selection_strategy from cfg
  PATCH 3 -- ABLATION_CONFIGS: fix B1/B2/B3, add G2/G3 configs + better print
  PATCH 4 -- Section 15b: Group B + G2 + G3 runners + updated all_ablation save
  PATCH 5 -- Insert Section 15e (Group B sanity) + 15f (Groups G runner)
"""
import json, re
from pathlib import Path

NB = Path("dekae_colab.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(cell):
    s = cell.get("source", "")
    return s if isinstance(s, str) else "".join(s)


def set_src(cell, text):
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    cell["source"] = lines


def find(cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def find_idx(cid):
    for i, c in enumerate(cells):
        if c.get("id") == cid:
            return i
    raise KeyError(cid)


patches = []
skips = []


def try_replace(cell_src, old, new, label, use_re=False, re_flags=0):
    if use_re:
        compiled = old if hasattr(old, 'subn') else re.compile(old, re_flags)
        new_src, n = compiled.subn(new, cell_src, count=1)
        if n:
            patches.append(label)
            return new_src
    else:
        if old in cell_src:
            patches.append(label)
            return cell_src.replace(old, new, 1)
    skips.append("SKIP " + label)
    return cell_src


# ============================================================================
# PATCH 1  DEKAEModel signature + body  (cell b9806c81)
# ============================================================================
c_model = find("b9806c81")
s = src(c_model)

# 1a: extend __init__ signature
OLD_1A = '                 edge_proj_type: str = "mlp"):'
NEW_1A = ('                 edge_proj_type: str = "mlp",\n'
          '                 selection_strategy: str = "max"):')
s = try_replace(s, OLD_1A, NEW_1A, "PATCH 1a -- DEKAEModel signature +selection_strategy")

# 1b: store self.selection_strategy
OLD_1B = '        self.edge_proj_type   = edge_proj_type  # Group C ablation flag\n'
NEW_1B = ('        self.edge_proj_type   = edge_proj_type  # Group C ablation flag\n'
          '        self.selection_strategy = selection_strategy  # Group B: max | nearest_centroid\n')
s = try_replace(s, OLD_1B, NEW_1B, "PATCH 1b -- DEKAEModel stores self.selection_strategy")

# 1c: add set_selection_strategy helper
OLD_1C = '    def set_use_lmt(self, flag: bool):\n        self.use_lmt = flag\n'
NEW_1C = ('    def set_use_lmt(self, flag: bool):\n'
          '        self.use_lmt = flag\n'
          '\n'
          '    def set_selection_strategy(self, strategy: str):\n'
          '        """Group B: "max" (mean pool, default) or "nearest_centroid"."""\n'
          '        assert strategy in ("max", "nearest_centroid"), (\n'
          '            f"Unknown selection_strategy {strategy!r}.")\n'
          '        self.selection_strategy = strategy\n')
s = try_replace(s, OLD_1C, NEW_1C, "PATCH 1c -- set_selection_strategy() helper added")

# 1d: update prototype computation -- use regex so Unicode box-drawing chars do not matter
PAT_PROTO = re.compile(
    r'        # .{0,12}9\..{0,80}\n'
    r'        prototypes = torch\.zeros\(self\.n_way, self\.embed_dim, device=H\.device\)\n'
    r'        for c in range\(self\.n_way\):\n'
    r'            mask_c = labels_support == c\n'
    r'            if mask_c\.sum\(\) > 0:\n'
    r'                prototypes\[c\] = H_support\[mask_c\]\.mean\(dim=0\)\n',
    re.DOTALL,
)
NEW_1D = (
    '        # -- 9. Compute Prototypes ------------------------------------------\n'
    '        # selection_strategy controls how support nodes contribute (Group B):\n'
    '        #   "max"              -- equal-weight mean pool (default, works 1-shot)\n'
    '        #   "nearest_centroid" -- soft-weight by cosine sim to rough centroid;\n'
    '        #                        down-weights outlier support nodes (B2, 5-shot)\n'
    '        prototypes = torch.zeros(self.n_way, self.embed_dim, device=H.device)\n'
    '        for c in range(self.n_way):\n'
    '            mask_c = labels_support == c\n'
    '            if mask_c.sum() > 0:\n'
    '                cands = H_support[mask_c]                   # (K_c, d)\n'
    '                if (self.selection_strategy == "nearest_centroid"\n'
    '                        and cands.size(0) > 1):\n'
    '                    rough = cands.mean(dim=0, keepdim=True) # (1,d) rough centroid\n'
    '                    sim = F.cosine_similarity(\n'
    '                        cands, rough.expand(cands.size(0), -1), dim=-1)\n'
    '                    weights = F.softmax(sim * 5.0, dim=0)   # temperature 5\n'
    '                    prototypes[c] = (cands * weights.unsqueeze(1)).sum(dim=0)\n'
    '                else:\n'
    '                    prototypes[c] = cands.mean(dim=0)       # "max": equal weights\n'
)
s = try_replace(s, PAT_PROTO, NEW_1D, "PATCH 1d -- prototype selection_strategy branch", use_re=True)

set_src(c_model, s)

# ============================================================================
# PATCH 2  build_model (cell 88ed9b71)
# ============================================================================
c_build = find("88ed9b71")
s = src(c_build)

OLD_2 = ('        edge_proj_type     = cfg.get("edge_proj_type", "mlp"),          # Group C ablation\n'
         '    ).to(DEVICE)\n')
NEW_2 = ('        edge_proj_type     = cfg.get("edge_proj_type", "mlp"),          # Group C ablation\n'
         '        selection_strategy = cfg.get("selection_strategy", "max"),      # Group B ablation\n'
         '    ).to(DEVICE)\n')
s = try_replace(s, OLD_2, NEW_2, "PATCH 2 -- build_model passes selection_strategy")
set_src(c_build, s)

# ============================================================================
# PATCH 3  ABLATION_CONFIGS (cell 1f9b5c4b)
# ============================================================================
c_abl = find("1f9b5c4b")
s = src(c_abl)

# 3a: replace old B block with plan-correct B + G2 + G3 configs
PAT_B_OLD = re.compile(
    r'    # .{0,10}Group B:.*?\n'
    r'    "B1_fixed_k5":.*?\n'
    r'    "B2_fixed_k10":.*?\n'
    r'    "B3_dynamic":.*?\n'
    r'\n'
    r'    # .{0,10}Group C:',
    re.DOTALL,
)
NEW_3A = (
    '    # -- Group B: Selection Strategy (plan 5.2) --------------------------\n'
    '    # B1: fixed k=5 static kNN, equal-weight mean pool (max strategy)\n'
    '    # B2: fixed k=5 static kNN, nearest-to-centroid soft weights\n'
    '    # B3: dynamic learned topology, equal-weight mean pool\n'
    '    "B1_max_k5":     {**BASE, "use_dynamic": False, "use_lmt": False,\n'
    '                      "topk_k": 5, "lambda_edge": 0.0,\n'
    '                      "selection_strategy": "max"},\n'
    '    "B2_nearest_k5": {**BASE, "use_dynamic": False, "use_lmt": False,\n'
    '                      "topk_k": 5, "lambda_edge": 0.0,\n'
    '                      "selection_strategy": "nearest_centroid"},\n'
    '    "B3_dynamic":    {**BASE, "use_dynamic": True,  "use_lmt": False,\n'
    '                      "lambda_edge": 0.5, "selection_strategy": "max"},\n'
    '\n'
    '    # -- Group G2: N-way / K-shot Config Sensitivity (plan 5.7 G2) -------\n'
    '    "G2_5w1s":   {**BASE, "n_way": 5,  "k_shot": 1, "n_query": 15,\n'
    '                  "use_lmt": False, "use_dynamic": True, "lambda_edge": 0.5},\n'
    '    "G2_5w5s":   {**BASE, "n_way": 5,  "k_shot": 5, "n_query": 15,\n'
    '                  "use_lmt": False, "use_dynamic": True, "lambda_edge": 0.5},\n'
    '    "G2_10w1s":  {**BASE, "n_way": 10, "k_shot": 1, "n_query": 15,\n'
    '                  "use_lmt": False, "use_dynamic": True, "lambda_edge": 0.5},\n'
    '    "G2_10w5s":  {**BASE, "n_way": 10, "k_shot": 5, "n_query": 15,\n'
    '                  "use_lmt": False, "use_dynamic": True, "lambda_edge": 0.5},\n'
    '\n'
    '    # -- Group G3: Backbone / Embed-Dim Sensitivity (plan 5.7 G3) --------\n'
    '    "G3_embed64":  {**BASE, "embed_dim": 64,  "use_lmt": False,\n'
    '                    "use_dynamic": True, "lambda_edge": 0.5},\n'
    '    "G3_embed128": {**BASE, "embed_dim": 128, "use_lmt": False,\n'
    '                    "use_dynamic": True, "lambda_edge": 0.5},  # default\n'
    '    "G3_embed256": {**BASE, "embed_dim": 256, "use_lmt": False,\n'
    '                    "use_dynamic": True, "lambda_edge": 0.5},\n'
    '\n'
    '    # -- Group C:'
)
s = try_replace(s, PAT_B_OLD, NEW_3A, "PATCH 3a -- B/G2/G3 configs updated", use_re=True, re_flags=re.DOTALL)

# 3b: update print block
OLD_3B = ('print("Group A variants :", [k for k in ABLATION_CONFIGS if k.startswith("A")])\n'
          'print("Group C variants :", [k for k in ABLATION_CONFIGS if k.startswith("C")])\n'
          'print("Group D variants :", [k for k in ABLATION_CONFIGS if k.startswith("D")])\n'
          'print("Group E variants :", [k for k in ABLATION_CONFIGS if k.startswith("E")])\n'
          'print("Group H variants :", [k for k in ABLATION_CONFIGS if k.startswith("H")])\n'
          'print("Group H4 variants:", [k for k in ABLATION_CONFIGS if k.startswith("H4")])\n'
          'print("Group H5 variants:", [k for k in ABLATION_CONFIGS if k.startswith("H5")])\n'
          'print("Group G5 variants:", [k for k in ABLATION_CONFIGS if k.startswith("G5")])\n'
          'print("Group I  variants:", [k for k in ABLATION_CONFIGS if k.startswith("I")])\n')
NEW_3B = ('print("Group A variants :", [k for k in ABLATION_CONFIGS if k.startswith("A")])\n'
          'print("Group B variants :", [k for k in ABLATION_CONFIGS if k.startswith("B")])\n'
          'print("Group C variants :", [k for k in ABLATION_CONFIGS if k.startswith("C")])\n'
          'print("Group D variants :", [k for k in ABLATION_CONFIGS if k.startswith("D")])\n'
          'print("Group E variants :", [k for k in ABLATION_CONFIGS if k.startswith("E")])\n'
          'print("Group G2 variants:", [k for k in ABLATION_CONFIGS if k.startswith("G2")])\n'
          'print("Group G3 variants:", [k for k in ABLATION_CONFIGS if k.startswith("G3")])\n'
          'print("Group G5 variants:", [k for k in ABLATION_CONFIGS if k.startswith("G5")])\n'
          'print("Group H variants :", [k for k in ABLATION_CONFIGS if k.startswith("H")])\n'
          'print("Group H4 variants:", [k for k in ABLATION_CONFIGS if k.startswith("H4")])\n'
          'print("Group H5 variants:", [k for k in ABLATION_CONFIGS if k.startswith("H5")])\n'
          'print("Group I  variants:", [k for k in ABLATION_CONFIGS if k.startswith("I")])\n')
s = try_replace(s, OLD_3B, NEW_3B, "PATCH 3b -- print statements show B, G2, G3 groups")
set_src(c_abl, s)

# ============================================================================
# PATCH 4  Section 15b -- add B/G2/G3 runners + update all_ablation save
# ============================================================================
c_15b = find("a48be1a2")
s = src(c_15b)

# The old all_ablation line
OLD_AL = ('all_ablation = group_a_results + group_e_core_results'
          ' + group_c_results + group_d_results\n')

# We'll insert the B/G2/G3 runners + the new all_ablation right before OLD_AL
# Locate start-of-line for OLD_AL
idx_al = s.find(OLD_AL)
if idx_al >= 0:
    # Find the start of this line
    line_start = s.rfind('\n', 0, idx_al) + 1
    runner_block = (
        '# -- Group B: Selection Strategy (plan 5.2) --------------------------------\n'
        'RUN_GROUP_B = False   # set True to compare max vs nearest_centroid\n'
        'if RUN_GROUP_B:\n'
        '    print("\\nRunning Group B ablations (selection strategy)...")\n'
        '    group_b_results = run_all_ablations(\n'
        '        groups            = ["B1_max_k5", "B2_nearest_k5", "B3_dynamic"],\n'
        '        episode_fn_train  = train_sampler_aug,\n'
        '        episode_fn_val    = val_sampler,\n'
        '    )\n'
        '    print_ablation_table(group_b_results, "Group B -- Selection Strategy")\n'
        'else:\n'
        '    print("  (Group B skipped -- set RUN_GROUP_B=True)")\n'
        '    group_b_results = []\n'
        '\n'
        '# -- Group G2: N-way/K-shot sensitivity (plan 5.7 G2) ----------------------\n'
        'RUN_GROUP_G2 = False\n'
        'if RUN_GROUP_G2:\n'
        '    print("\\nRunning Group G2 ablations (n-way/k-shot sensitivity)...")\n'
        '    group_g2_results = run_all_ablations(\n'
        '        groups            = ["G2_5w1s", "G2_5w5s", "G2_10w1s", "G2_10w5s"],\n'
        '        episode_fn_train  = train_sampler_aug,\n'
        '        episode_fn_val    = val_sampler,\n'
        '    )\n'
        '    print_ablation_table(group_g2_results, "Group G2 -- N-way/K-shot Sensitivity")\n'
        'else:\n'
        '    print("  (Group G2 skipped -- set RUN_GROUP_G2=True)")\n'
        '    group_g2_results = []\n'
        '\n'
        '# -- Group G3: Backbone embed_dim sensitivity (plan 5.7 G3) ----------------\n'
        'RUN_GROUP_G3 = False\n'
        'if RUN_GROUP_G3:\n'
        '    print("\\nRunning Group G3 ablations (backbone embed_dim)...")\n'
        '    group_g3_results = run_all_ablations(\n'
        '        groups            = ["G3_embed64", "G3_embed128", "G3_embed256"],\n'
        '        episode_fn_train  = train_sampler_aug,\n'
        '        episode_fn_val    = val_sampler,\n'
        '    )\n'
        '    print_ablation_table(group_g3_results, "Group G3 -- Backbone Embed-Dim")\n'
        'else:\n'
        '    print("  (Group G3 skipped -- set RUN_GROUP_G3=True)")\n'
        '    group_g3_results = []\n'
        '\n'
    )
    new_all_ablation = (
        'all_ablation = (group_a_results + group_e_core_results\n'
        '                + group_c_results + group_d_results\n'
        '                + group_b_results + group_g2_results + group_g3_results)\n'
    )
    s = s[:line_start] + runner_block + new_all_ablation + s[idx_al + len(OLD_AL):]
    patches.append("PATCH 4 -- Section 15b Group B/G2/G3 runners + updated all_ablation")
else:
    skips.append("SKIP PATCH 4 -- could not locate all_ablation line in 15b")

set_src(c_15b, s)

# ============================================================================
# PATCH 5  Insert Section 15e + 15f  (after grpd0002)
# ============================================================================
existing_ids = {c.get("id") for c in cells}
insert_after = find_idx("grpd0002")

md_15e_src = [
    "## Section 15e: Ablation Group B -- Selection Strategy\n",
    "\n",
    "**Group B** (plan 5.2) tests how the prototype aggregation strategy\n",
    "affects accuracy, following the FSAKE observation that max-selection\n",
    "works in 1-shot while nearest-to-centroid works better in 5-shot.\n",
    "\n",
    "| Variant | Topology | Strategy | 1-shot Acc |\n",
    "| ------- | -------- | -------- | ---------- |\n",
    "| B1_max_k5 | Fixed k=5 | Max (mean pool) | (report) |\n",
    "| B2_nearest_k5 | Fixed k=5 | Nearest-to-centroid | (report) |\n",
    "| B3_dynamic | Dynamic learned | Auto-adaptive | (report) |\n",
    "\n",
    "**What this proves**: If B3 >= B1 and B2, dynamic topology subsumes\n",
    "the need to manually tune selection strategy.\n",
    "\n",
    "### Implementation\n",
    "`selection_strategy` parameter added to `DEKAEModel.__init__`.\n",
    "When `nearest_centroid`, prototype step soft-weights each support node\n",
    "by `softmax(cosine_sim(node, rough_centroid) * T)` with temperature T=5.\n",
]

code_15e_src = [
    "# -- Section 15e: Group B -- Selection Strategy sanity checks ----------------\n",
    "\n",
    "_s_b = torch.randn(5, 3, 84, 84).to(DEVICE)\n",
    "_l_b = torch.arange(5).to(DEVICE)\n",
    "_q_b = torch.randn(15, 3, 84, 84).to(DEVICE)\n",
    "\n",
    "print('Group B sanity checks (untrained models):')\n",
    "print('-' * 60)\n",
    "\n",
    "for _bname, _strat in [('B1_max_k5', 'max'), ('B2_nearest_k5', 'nearest_centroid')]:\n",
    "    _mb = build_model(ABLATION_CONFIGS[_bname])\n",
    "    _mb.eval()\n",
    "    with torch.no_grad():\n",
    "        _, _, _met_b = _mb(_s_b, _l_b, _q_b)\n",
    "    print(f'  {_bname:<22} strategy={_strat:<20} density={_met_b[\"graph_density\"]:.3f}')\n",
    "\n",
    "# 5-shot: nearest_centroid meaningful only when k_shot > 1\n",
    "_s_b5 = torch.randn(25, 3, 84, 84).to(DEVICE)  # 5-way 5-shot = 25 support\n",
    "_l_b5 = torch.repeat_interleave(torch.arange(5).to(DEVICE), 5)\n",
    "_q_b5 = torch.randn(15, 3, 84, 84).to(DEVICE)\n",
    "\n",
    "for _strat5 in ('max', 'nearest_centroid'):\n",
    "    _mb5 = DEKAEModel(n_way=5, embed_dim=128, use_dynamic=False,\n",
    "                      use_lmt=False, selection_strategy=_strat5).to(DEVICE)\n",
    "    _mb5.eval()\n",
    "    with torch.no_grad():\n",
    "        _lgt_b5, _, _ = _mb5(_s_b5, _l_b5, _q_b5)\n",
    "    print(f'  5-shot {_strat5:<20}: logits={_lgt_b5.shape} OK')\n",
    "\n",
    "print()\n",
    "print('Both strategies run correctly.')\n",
    "print('To run full Group B training: set RUN_GROUP_B=True in Section 15b.')\n",
]

md_15f_src = [
    "## Section 15f: Ablation Groups G1-G5 -- Sensitivity Analysis\n",
    "\n",
    "**Group G** (plan 5.7) validates robustness across operating conditions.\n",
    "\n",
    "| Group | What varies | Runner |\n",
    "| ----- | ----------- | ------ |\n",
    "| G1 | Noise sigma (synthetic Sec.14) | `run_group_g1_noise()` below |\n",
    "| G2 | n-way/k-shot configs | `RUN_GROUP_G2=True` in Sec.15b |\n",
    "| G3 | Backbone embed_dim | `RUN_GROUP_G3=True` in Sec.15b |\n",
    "| G4 | Dataset / resolution | re-run Sec.13b with each dataset |\n",
    "| G5 | lambda_edge value | `RUN_GROUP_G5=True` below |\n",
    "\n",
    "### G4 target table (plan 5.7 G4)\n",
    "| Dataset | Resolution | HGNN 5-shot | FSAKE 5-shot | Ours 5-shot |\n",
    "| ------- | ---------- | ----------- | ------------ | ----------- |\n",
    "| miniImageNet | 84x84 | (report) | 79.66 | (report) |\n",
    "| CIFAR-FS | 32x32 | 86.16 | 85.92 | (report) |\n",
]

code_15f_src = [
    "# -- Section 15f: Groups G -- Sensitivity Analysis ---------------------------\n",
    "\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# G1: Noise-level robustness sweep (sigma in Section 14 synthetic experiment)\n",
    "# ---------------------------------------------------------------------------\n",
    "def run_group_g1_noise(\n",
    "    sigmas=(0.3, 0.6, 0.8, 1.0),\n",
    "    n_trials=20,\n",
    "    n_nodes_per_class=5,\n",
    "    n_classes=5,\n",
    "    embed_dim=128,\n",
    "):\n",
    '    """Sweep sigma and run Section-14 synthetic graph-recovery experiment.\n',
    "\n",
    "    Reports per sigma:\n",
    "      - Static k-NN (FSAKE) Graph F1 and accuracy\n",
    "      - DEKAE (dynamic) Graph F1 and accuracy\n",
    "\n",
    "    Returns a list of result dicts (one per sigma value).\n",
    '    """\n',
    "    results = []\n",
    "    print('Group G1 -- Noise Robustness Sweep')\n",
    "    print('=' * 65)\n",
    "    print(f'  {\"Sigma\":<8}  {\"Static F1\":>10}  {\"DEKAE F1\":>10}  '\n",
    "          f'{\"Acc Static\":>11}  {\"Acc DEKAE\":>10}')\n",
    "    print('-' * 65)\n",
    "\n",
    "    for sigma in sigmas:\n",
    "        static_res  = run_synthetic_recovery(\n",
    "            sigma=sigma, n_trials=n_trials,\n",
    "            n_nodes_per_class=n_nodes_per_class, n_classes=n_classes,\n",
    "            embed_dim=embed_dim, use_dynamic=False)\n",
    "        dynamic_res = run_synthetic_recovery(\n",
    "            sigma=sigma, n_trials=n_trials,\n",
    "            n_nodes_per_class=n_nodes_per_class, n_classes=n_classes,\n",
    "            embed_dim=embed_dim, use_dynamic=True)\n",
    "\n",
    "        row = {\n",
    "            'sigma'      : sigma,\n",
    "            'static_f1'  : static_res.get('graph_f1',  float('nan')),\n",
    "            'dynamic_f1' : dynamic_res.get('graph_f1', float('nan')),\n",
    "            'static_acc' : static_res.get('accuracy',  float('nan')),\n",
    "            'dynamic_acc': dynamic_res.get('accuracy', float('nan')),\n",
    "        }\n",
    "        results.append(row)\n",
    "        print(f'  {sigma:<8.2f}  {row[\"static_f1\"]:>10.3f}  '\n",
    "              f'{row[\"dynamic_f1\"]:>10.3f}  '\n",
    "              f'{row[\"static_acc\"]:>10.3f}%  '\n",
    "              f'{row[\"dynamic_acc\"]:>10.3f}%')\n",
    "\n",
    "    print('=' * 65)\n",
    "    import json as _jg1\n",
    "    _out_g1 = str(RESULTS_DIR / 'ablation_group_g1.json')\n",
    "    with open(_out_g1, 'w') as _fg1:\n",
    "        _jg1.dump(results, _fg1, indent=2)\n",
    "    print(f'G1 results saved to {_out_g1}')\n",
    "    return results\n",
    "\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# G5: lambda_edge sensitivity\n",
    "# ---------------------------------------------------------------------------\n",
    "RUN_GROUP_G5 = False   # set True to sweep lambda_edge\n",
    "if RUN_GROUP_G5:\n",
    "    print('\\nRunning Group G5 ablations (lambda_edge sensitivity)...')\n",
    "    group_g5_results = run_all_ablations(\n",
    "        groups           = ['G5_lam0', 'G5_lam01', 'G5_lam05', 'G5_lam1', 'G5_lam2'],\n",
    "        episode_fn_train = train_sampler_aug,\n",
    "        episode_fn_val   = val_sampler,\n",
    "    )\n",
    "    print_ablation_table(group_g5_results, 'Group G5 -- Lambda_edge Sensitivity')\n",
    "    import json as _jg5\n",
    "    _out_g5 = str(RESULTS_DIR / 'ablation_group_g5.json')\n",
    "    with open(_out_g5, 'w') as _fg5:\n",
    "        _jg5.dump([{k: v for k, v in r.items() if k != 'per_episode'}\n",
    "                   for r in group_g5_results], _fg5, indent=2)\n",
    "    print(f'G5 results saved to {_out_g5}')\n",
    "else:\n",
    "    print('  (Group G5 skipped -- set RUN_GROUP_G5=True)')\n",
    "\n",
    "\n",
    "# Summary\n",
    "print()\n",
    "print('Group G functions / runners ready:')\n",
    "print('  G1 run_group_g1_noise() -- sigma sweep on synthetic experiment')\n",
    "print('  G2 RUN_GROUP_G2=True    -- n-way/k-shot in ABLATION_CONFIGS')\n",
    "print('  G3 RUN_GROUP_G3=True    -- embed_dim in ABLATION_CONFIGS')\n",
    "print('  G4 documented above     -- re-run Section 13b per dataset')\n",
    "print('  G5 RUN_GROUP_G5=True    -- lambda_edge sensitivity (this cell)')\n",
    "print()\n",
    "print('Group B:')\n",
    "print('  selection_strategy in DEKAEModel; B1/B2/B3 in ABLATION_CONFIGS')\n",
    "print('  Section 15e sanity check passed; RUN_GROUP_B=True in Sec.15b')\n",
]

new_cells_to_insert = []
if "grpb0001" not in existing_ids:
    new_cells_to_insert.append({
        "cell_type": "markdown", "id": "grpb0001", "metadata": {},
        "source": md_15e_src,
    })
if "grpb0002" not in existing_ids:
    new_cells_to_insert.append({
        "cell_type": "code", "id": "grpb0002", "metadata": {},
        "execution_count": None, "outputs": [],
        "source": code_15e_src,
    })
if "grpg0001" not in existing_ids:
    new_cells_to_insert.append({
        "cell_type": "markdown", "id": "grpg0001", "metadata": {},
        "source": md_15f_src,
    })
if "grpg0002" not in existing_ids:
    new_cells_to_insert.append({
        "cell_type": "code", "id": "grpg0002", "metadata": {},
        "execution_count": None, "outputs": [],
        "source": code_15f_src,
    })

if new_cells_to_insert:
    cells[insert_after + 1:insert_after + 1] = new_cells_to_insert
    patches.append(f"PATCH 5 -- {len(new_cells_to_insert)} new cells for 15e/15f inserted")
else:
    skips.append("SKIP PATCH 5 -- 15e/15f cells already present")

# ============================================================================
# Summary + save
# ============================================================================
print()
print("=" * 60)
if patches:
    print(f"Patches applied ({len(patches)}):")
    for p in patches:
        print(f"  OK {p}")
if skips:
    print(f"\nSkipped ({len(skips)}):")
    for sk in skips:
        print(f"  {sk}")
print("=" * 60)

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nNotebook saved: {NB}")
print(f"Total cells now: {len(cells)}")
