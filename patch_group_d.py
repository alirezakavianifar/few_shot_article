"""
Patch dekae_colab.ipynb — Group D: Seed Stability (plan §5.4 + Risk 8).

Changes:
  1. TOC cell          — add Group D row
  2. ABLATION_CONFIGS  — add D_seed* entries (5 seeds × full model)
  3. Section 15b       — add Group D runner + group_d_results to all_ablation save
  4. New cells 15d (md + code):
       - run_seed_stability()        → trains full model N times, reports mean±std±CI
       - topology_init_sensitivity() → same episode, vary MLP init, Frobenius norm
       - Section 15d sanity check
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "dekae_colab.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]


def by_id(cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    raise KeyError(f"Cell id not found: {cid!r}")


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    lines = text.splitlines(keepends=True)
    # ensure last line has no trailing newline (notebook convention)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    cell["source"] = lines


patches = []

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — TOC: insert Group D row
# ═══════════════════════════════════════════════════════════════════════════════
toc = by_id("4e915e4e")
OLD = "| 15c | Ablation Group C — Edge Feature Comparison |"
NEW = ("| 15c | Ablation Group C — Edge Feature Comparison |\n"
       "| 15d | Ablation Group D — Seed Stability & Topology Init Sensitivity |")
if OLD in src(toc):
    set_src(toc, src(toc).replace(OLD, NEW))
    patches.append("PATCH 1 — TOC updated with Group D")
else:
    print("SKIP PATCH 1 (already applied)")

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — ABLATION_CONFIGS: add Group D entries (cell 1f9b5c4b)
# ═══════════════════════════════════════════════════════════════════════════════
abl = by_id("1f9b5c4b")

OLD_G5 = (
    "    # ── Group G5: Lambda_edge Sensitivity (plan §5.7 G5) ─────────────────────\n"
    "    # NOTE: Implementation names kept as G4 for continuity, but these correspond\n"
    "    # to plan §5.7 G5 (Edge Loss Weight Sensitivity)."
)
NEW_D_G5 = """\
    # ── Group D: Seed Stability (plan §5.4) ──────────────────────────────────────
    # Run the full DEKAE model (= A5_DEKAE_noLMT config) with 5 different random
    # seeds.  Each variant is identical in all hyperparameters — only the seed
    # changes.  Collect per-seed test accuracy and topology stats; report mean±std
    # and 95% CI across seeds.
    #
    # Low std across seeds → training is stable and reproducible.
    # High std → dynamic adjacency is sensitive to initialization (see Risk 4/8).
    #
    # run_seed_stability() (defined in Section 15d) is the recommended runner;
    # these config entries also allow individual seeds via run_ablation() if needed.
    #
    # Seeds chosen: 42(default), 0, 1, 7, 123  — spread across the low-integer
    # range which is standard in FSL reproducibility ablations.
    "D_seed42":  {**BASE, "use_dynamic": True, "use_edge_proj": True,
                  "use_case2": True, "use_lmt": False, "lambda_edge": 0.5,
                  "seed": 42},
    "D_seed0":   {**BASE, "use_dynamic": True, "use_edge_proj": True,
                  "use_case2": True, "use_lmt": False, "lambda_edge": 0.5,
                  "seed": 0},
    "D_seed1":   {**BASE, "use_dynamic": True, "use_edge_proj": True,
                  "use_case2": True, "use_lmt": False, "lambda_edge": 0.5,
                  "seed": 1},
    "D_seed7":   {**BASE, "use_dynamic": True, "use_edge_proj": True,
                  "use_case2": True, "use_lmt": False, "lambda_edge": 0.5,
                  "seed": 7},
    "D_seed123": {**BASE, "use_dynamic": True, "use_edge_proj": True,
                  "use_case2": True, "use_lmt": False, "lambda_edge": 0.5,
                  "seed": 123},

    # ── Group G5: Lambda_edge Sensitivity (plan §5.7 G5) ─────────────────────────
    # NOTE: Implementation names kept as G4 for continuity, but these correspond
    # to plan §5.7 G5 (Edge Loss Weight Sensitivity)."""

if OLD_G5 in src(abl):
    set_src(abl, src(abl).replace(OLD_G5, NEW_D_G5))
    patches.append("PATCH 2 — ABLATION_CONFIGS: Group D seeds added")
else:
    print("SKIP PATCH 2 (already applied)")

# Update print statements at the bottom of the ablation config cell
OLD_PRINT = 'print("Group C variants :", [k for k in ABLATION_CONFIGS if k.startswith("C")])'
NEW_PRINT = (
    'print("Group C variants :", [k for k in ABLATION_CONFIGS if k.startswith("C")])\n'
    'print("Group D variants :", [k for k in ABLATION_CONFIGS if k.startswith("D")])'
)
if OLD_PRINT in src(abl):
    set_src(abl, src(abl).replace(OLD_PRINT, NEW_PRINT))
    patches.append("PATCH 2b — ABLATION_CONFIGS print updated")
else:
    print("SKIP PATCH 2b (already applied)")

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — Section 15b: add Group D runner + include in all_ablation save
# ═══════════════════════════════════════════════════════════════════════════════
exec15b = by_id("a48be1a2")

# Insert Group D runner block before the final save
OLD_SAVE = (
    "# ── Save all ablation results to Drive ────────────────────────────────────────\n"
    "all_ablation = group_a_results + group_e_core_results + group_c_results"
)
NEW_D_SAVE = """\
# ── Group D: Seed Stability (plan §5.4) ──────────────────────────────────────
# Trains the full DEKAE model with 5 different seeds; reports mean/std/CI.
# Uses the dedicated run_seed_stability() function defined in Section 15d
# (which is more efficient than run_all_ablations for this specific purpose).
# Set RUN_GROUP_D=True to execute (each seed requires ~60–90 min on T4).
RUN_GROUP_D = False   # ← set True when you have ~6–8 hours available

if RUN_GROUP_D:
    print("\\nRunning Group D — seed stability (5 seeds)…")
    group_d_summary, group_d_results = run_seed_stability(
        cfg              = ABLATION_CONFIGS["D_seed42"],   # base config (seed overridden)
        seeds            = [42, 0, 1, 7, 123],
        episode_fn_train = train_sampler_aug,
        episode_fn_val   = val_sampler,
        episode_fn_test  = test_sampler,
    )
    print("\\nGroup D — Seed Stability Summary:")
    print(f"  Accuracy : {group_d_summary['mean_acc']*100:.2f}% "
          f"± {group_d_summary['std_acc']*100:.2f}%  "
          f"(95% CI: ±{group_d_summary['ci_95']*100:.2f}%)")
    print(f"  Density  : {group_d_summary['mean_density']:.3f} "
          f"± {group_d_summary['std_density']:.3f}")
    print(f"  Topo stab: mean Frobenius norm across init seeds = "
          f"{group_d_summary['topo_init_frob_mean']:.4f} "
          f"± {group_d_summary['topo_init_frob_std']:.4f}")
else:
    print("  (Group D skipped — set RUN_GROUP_D=True after Group A completes)")
    group_d_results = []

# ── Save all ablation results to Drive ────────────────────────────────────────
all_ablation = group_a_results + group_e_core_results + group_c_results + group_d_results"""

if OLD_SAVE in src(exec15b):
    set_src(exec15b, src(exec15b).replace(OLD_SAVE, NEW_D_SAVE))
    patches.append("PATCH 3 — Section 15b: Group D runner added")
else:
    print("SKIP PATCH 3 (already applied)")

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 4 — Insert new Section 15d cells (after cell grpc0002 = index 39)
# ═══════════════════════════════════════════════════════════════════════════════
already_d = any(c.get("id") in ("grpd0001", "grpd0002") for c in cells)

if not already_d:
    insert_after = next(i for i, c in enumerate(cells) if c.get("id") == "grpc0002")

    md_15d = {
        "cell_type": "markdown",
        "id": "grpd0001",
        "metadata": {},
        "source": [
            "## Section 15d: Ablation Group D — Seed Stability & Topology Init Sensitivity\n",
            "\n",
            "**Group D** (plan §5.4 + Risk 8) validates reproducibility along two axes:\n",
            "\n",
            "### D1 — Across-Run Seed Stability\n",
            "Trains the full DEKAE model (same config as `A5_DEKAE_noLMT`) with 5 different\n",
            "global random seeds `{42, 0, 1, 7, 123}`. Reports:\n",
            "- Per-seed test accuracy, graph density, and avg degree\n",
            "- **Mean ± std** and **95% CI** across seeds\n",
            "- Low std (< 0.5%) → model is robust to initialization; high std → unstable\n",
            "\n",
            "| Seed | Accuracy | Density | Avg Degree |\n",
            "|------|----------|---------|------------|\n",
            "| 42   | (report) | (report)| (report)   |\n",
            "| 0    | (report) | (report)| (report)   |\n",
            "| 1    | (report) | (report)| (report)   |\n",
            "| 7    | (report) | (report)| (report)   |\n",
            "| 123  | (report) | (report)| (report)   |\n",
            "| **Mean±std** | (report) | (report)| (report) |\n",
            "\n",
            "### D2 — Topology Initialization Sensitivity (Risk 8 mitigation)\n",
            "Fixes the backbone weights (trained checkpoint) and re-initializes *only* the\n",
            "edge scoring MLP (`DynamicTopologyModule.W1`, `W2`, `edge_scorer`) with\n",
            "`n_inits=5` different seeds. For each init, runs `n_episodes` episodes and\n",
            "records the recovered adjacency $A'$. Reports the **mean Frobenius norm**\n",
            "$\\\\|A'_{\\\\text{init}_i} - A'_{\\\\text{init}_j}\\\\|_F$ across pairs:\n",
            "- Low variance → near-unique convergence (validates Risk 8 identifiability claim)\n",
            "- High variance → topology is sensitive to MLP init (add this as a limitation)\n",
        ],
        "outputs": [],
        "execution_count": None,
    }

    code_15d = {
        "cell_type": "code",
        "id": "grpd0002",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# -- Section 15d: Group D -- Seed Stability & Topology Init Sensitivity ----\n",
            "import copy, itertools\n",
            "\n",
            "# ---------------------------------------------------------------------------\n",
            "# D1: run_seed_stability()\n",
            "#   Trains the full model with multiple global seeds.\n",
            "#   Returns a summary dict and a list of per-seed result dicts.\n",
            "# ---------------------------------------------------------------------------\n",
            "def run_seed_stability(cfg,\n",
            "                       seeds,\n",
            "                       episode_fn_train,\n",
            "                       episode_fn_val,\n",
            "                       episode_fn_test=None,\n",
            "                       n_test_episodes=600):\n",
            '    """\n',
            "    Train the DEKAE model once per seed and collect accuracy + topology metrics.\n",
            "\n",
            "    Parameters\n",
            "    ----------\n",
            "    cfg              : base config dict (seed key will be overridden per run)\n",
            "    seeds            : list of integer seeds to sweep\n",
            "    episode_fn_train : callable for training episodes\n",
            "    episode_fn_val   : callable for validation episodes\n",
            "    episode_fn_test  : callable for test episodes (optional; uses val if None)\n",
            "    n_test_episodes  : number of episodes for final accuracy estimate per seed\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    summary : dict with mean/std/ci_95 across seeds for accuracy + graph metrics\n",
            "    results : list of per-seed dicts\n",
            '    """\n',
            "    test_fn = episode_fn_test if episode_fn_test is not None else episode_fn_val\n",
            "    per_seed = []\n",
            "\n",
            "    for seed in seeds:\n",
            "        seed_cfg = copy.deepcopy(cfg)\n",
            "        seed_cfg['seed'] = seed\n",
            "        run_name = f'D_seed{seed}'\n",
            "        print(f'\\n{\"=\"*55}')\n",
            "        print(f'  Group D -- seed={seed}  ({seeds.index(seed)+1}/{len(seeds)})')\n",
            "        print(f'{\"=\"*55}')\n",
            "\n",
            "        trained, history = train(\n",
            "            seed_cfg, episode_fn_train, episode_fn_val,\n",
            "            run_name=run_name, log_wandb=False\n",
            "        )\n",
            "        res = evaluate(trained, test_fn,\n",
            "                       seed_cfg['n_way'], seed_cfg['k_shot'], seed_cfg['n_query'],\n",
            "                       n_episodes=n_test_episodes)\n",
            "\n",
            "        # Topology stability: std of graph density across test episodes\n",
            "        topo_std = float(np.std(res.get('per_episode_density',\n",
            "                                        [res['avg_density']] * n_test_episodes)))\n",
            "\n",
            "        per_seed.append({\n",
            "            'seed'        : seed,\n",
            "            'acc'         : res['mean_acc'],\n",
            "            'ci_95'       : res['ci_95'],\n",
            "            'avg_density' : res['avg_density'],\n",
            "            'avg_degree'  : res['avg_degree'],\n",
            "            'topo_std'    : topo_std,\n",
            "        })\n",
            "        print(f'  Seed {seed}: acc={res[\"mean_acc\"]*100:.2f}%  '\n",
            "              f'density={res[\"avg_density\"]:.3f}  topo_std={topo_std:.4f}')\n",
            "\n",
            "    accs      = [r['acc']         for r in per_seed]\n",
            "    densities = [r['avg_density'] for r in per_seed]\n",
            "\n",
            "    # Run D2 topology-init sensitivity using the last trained model's backbone\n",
            "    frob_mean, frob_std = topology_init_sensitivity(\n",
            "        backbone    = trained.backbone,\n",
            "        edge_module = trained.edge_modules[0],\n",
            "        dyn_module  = trained.dynamic_modules[0],\n",
            "        episode_fn  = test_fn,\n",
            "        n_episodes  = 30,\n",
            "        n_inits     = 5,\n",
            "        knn_k       = cfg['topk_k'],\n",
            "        embed_dim   = cfg['embed_dim'],\n",
            "        rank        = cfg['rank'],\n",
            "    )\n",
            "\n",
            "    summary = {\n",
            "        'seeds'              : seeds,\n",
            "        'mean_acc'           : float(np.mean(accs)),\n",
            "        'std_acc'            : float(np.std(accs)),\n",
            "        'ci_95'              : 1.96 * float(np.std(accs)) / (len(accs) ** 0.5),\n",
            "        'mean_density'       : float(np.mean(densities)),\n",
            "        'std_density'        : float(np.std(densities)),\n",
            "        'topo_init_frob_mean': frob_mean,\n",
            "        'topo_init_frob_std' : frob_std,\n",
            "    }\n",
            "\n",
            "    # Print summary table\n",
            "    print(f'\\n{\"=\"*65}')\n",
            "    print(f'  Group D -- Seed Stability Summary ({len(seeds)} seeds)')\n",
            "    print(f'{\"=\"*65}')\n",
            "    print(f'  {\"Seed\":<8} {\"Acc (%)\":>9}  {\"CI95\":>6}  {\"Density\":>8}  {\"Avg Deg\":>8}')\n",
            "    print(f'  {\"-\"*50}')\n",
            "    for r in per_seed:\n",
            "        print(f'  {r[\"seed\"]:<8} {r[\"acc\"]*100:>8.2f}%  '\n",
            "              f'+-{r[\"ci_95\"]*100:>4.2f}%  '\n",
            "              f'{r[\"avg_density\"]:>8.3f}  {r[\"avg_degree\"]:>8.2f}')\n",
            "    print(f'  {\"-\"*50}')\n",
            "    print(f'  {\"Mean+-std\":<8} {summary[\"mean_acc\"]*100:>8.2f}%  '\n",
            "          f'+-{summary[\"std_acc\"]*100:>4.2f}%  '\n",
            "          f'{summary[\"mean_density\"]:>8.3f}')\n",
            "    print(f'  {\"95% CI\":<8} +-{summary[\"ci_95\"]*100:.2f}%')\n",
            "    print(f'\\n  Topology Init Sensitivity (D2):')\n",
            "    print(f'    Mean Frobenius norm across init pairs: '\n",
            "          f'{summary[\"topo_init_frob_mean\"]:.4f} +- {summary[\"topo_init_frob_std\"]:.4f}')\n",
            "    print(f'    (Near zero = near-unique topology convergence; validates Risk 8)')\n",
            "    print(f'{\"=\"*65}')\n",
            "\n",
            "    # Save to Drive\n",
            "    with open(str(RESULTS_DIR / 'ablation_group_d.json'), 'w') as _f:\n",
            "        json.dump({'summary': summary, 'per_seed': per_seed}, _f, indent=2)\n",
            "    print(f'\\nGroup D results saved -> {RESULTS_DIR}/ablation_group_d.json')\n",
            "\n",
            "    return summary, per_seed\n",
            "\n",
            "\n",
            "# ---------------------------------------------------------------------------\n",
            "# D2: topology_init_sensitivity()\n",
            "#   Fixes the backbone; re-inits only edge MLP weights; measures Frobenius\n",
            "#   norm between recovered adjacency matrices across init seeds.\n",
            "#   Low norm variance -> near-unique convergence (plan section 1.5 Risk 8).\n",
            "# ---------------------------------------------------------------------------\n",
            "def topology_init_sensitivity(backbone, edge_module, dyn_module,\n",
            "                               episode_fn, n_episodes=30,\n",
            "                               n_inits=5, knn_k=5,\n",
            "                               embed_dim=128, rank=16):\n",
            '    """\n',
            "    For each of n_inits random initializations of the edge scoring weights,\n",
            "    run n_episodes episodes using the FIXED backbone and collect the recovered\n",
            "    adjacency A_prime. Report pairwise Frobenius norms between adjacency matrices.\n",
            "\n",
            "    The backbone weights are kept frozen; only DynamicTopologyModule and\n",
            "    EdgeIncidenceModule weights are re-initialized per run. This isolates\n",
            "    topology sensitivity from representation sensitivity.\n",
            "\n",
            "    Returns\n",
            "    -------\n",
            "    frob_mean : float  mean pairwise Frobenius norm across init seed pairs\n",
            "    frob_std  : float  std of pairwise Frobenius norms\n",
            '    """\n',
            "    backbone.eval()\n",
            "    backbone.to(DEVICE)\n",
            "\n",
            "    adj_lists = []   # one mean adjacency per init\n",
            "\n",
            "    for init_seed in range(n_inits):\n",
            "        torch.manual_seed(init_seed)\n",
            "        # Fresh copies with new random weights\n",
            "        eim_fresh = EdgeIncidenceModule(embed_dim, embed_dim, hidden=64).to(DEVICE)\n",
            "        dtm_fresh = DynamicTopologyModule(embed_dim, rank, embed_dim).to(DEVICE)\n",
            "\n",
            "        # Explicit re-init to ensure independence from global state\n",
            "        for mod in [eim_fresh, dtm_fresh]:\n",
            "            for p in mod.parameters():\n",
            "                if p.dim() >= 2:\n",
            "                    nn.init.xavier_uniform_(p)\n",
            "                else:\n",
            "                    nn.init.zeros_(p)\n",
            "\n",
            "        eim_fresh.eval(); dtm_fresh.eval()\n",
            "        ep_adjs = []\n",
            "\n",
            "        with torch.no_grad():\n",
            "            for _ in range(n_episodes):\n",
            "                s_imgs, s_lbl, q_imgs, q_lbl = episode_fn(5, 1, 15)\n",
            "                all_imgs = torch.cat([s_imgs, q_imgs], dim=0).to(DEVICE)\n",
            "                H = backbone(all_imgs)                        # fixed backbone\n",
            "                A_init = build_knn_adjacency(H, k=knn_k)\n",
            "                E_f, src_f, dst_f = eim_fresh(H, A_init)\n",
            "                A_prime, _ = dtm_fresh(H, E_f, src_f, dst_f, sparsity_reg=False)\n",
            "                ep_adjs.append(A_prime.cpu())\n",
            "\n",
            "        # Mean adjacency over episodes (reduces episode noise)\n",
            "        mean_adj = torch.stack(ep_adjs).mean(dim=0)\n",
            "        adj_lists.append(mean_adj)\n",
            "\n",
            "    # Pairwise Frobenius norms\n",
            "    frob_norms = []\n",
            "    for i, j in itertools.combinations(range(n_inits), 2):\n",
            "        frob = (adj_lists[i] - adj_lists[j]).norm(p='fro').item()\n",
            "        frob_norms.append(frob)\n",
            "\n",
            "    frob_mean = float(np.mean(frob_norms)) if frob_norms else 0.0\n",
            "    frob_std  = float(np.std(frob_norms))  if frob_norms else 0.0\n",
            "\n",
            "    print(f'  D2 Topology Init Sensitivity: {n_inits} inits x {n_episodes} episodes')\n",
            "    print(f'    Pairwise Frobenius norms: {[round(f,4) for f in frob_norms]}')\n",
            "    print(f'    Mean={frob_mean:.4f}  Std={frob_std:.4f}')\n",
            "    if frob_mean < 0.05:\n",
            "        print('    -> Low variance: topology converges near-uniquely (Risk 8 satisfied)')\n",
            "    else:\n",
            "        print('    -> Moderate/high variance: topology sensitive to MLP init')\n",
            "\n",
            "    return frob_mean, frob_std\n",
            "\n",
            "\n",
            "# ---------------------------------------------------------------------------\n",
            "# Quick sanity check (runs without training)\n",
            "# ---------------------------------------------------------------------------\n",
            "print('Group D functions defined:')\n",
            "print('  run_seed_stability()        -- D1: trains full model N seeds, reports mean+-std+-CI')\n",
            "print('  topology_init_sensitivity() -- D2: fixes backbone, varies MLP init, Frobenius norm')\n",
            "print()\n",
            "print('Quick D2 sanity check (untrained backbone)...')\n",
            "_bb_dummy  = Conv4(embed_dim=128).to(DEVICE)\n",
            "_eim_dummy = EdgeIncidenceModule(128, 128, hidden=64).to(DEVICE)\n",
            "_dtm_dummy = DynamicTopologyModule(128, 16, 128).to(DEVICE)\n",
            "\n",
            "_frob_m, _frob_s = topology_init_sensitivity(\n",
            "    backbone    = _bb_dummy,\n",
            "    edge_module = _eim_dummy,\n",
            "    dyn_module  = _dtm_dummy,\n",
            "    episode_fn  = _dummy_episode_fn,   # Section 13 dummy function\n",
            "    n_episodes  = 5,\n",
            "    n_inits     = 3,\n",
            "    knn_k       = 5,\n",
            "    embed_dim   = 128,\n",
            "    rank        = 16,\n",
            ")\n",
            "print(f'Untrained D2 sanity: frob_mean={_frob_m:.4f}  frob_std={_frob_s:.4f}')\n",
            "print('(High frob expected for untrained model -- meaningful only post-training)')\n",
            "print()\n",
            "print('To run full Group D: set RUN_GROUP_D=True in Section 15b.')\n",
        ],
    }

    cells.insert(insert_after + 1, md_15d)
    cells.insert(insert_after + 2, code_15d)
    patches.append("PATCH 4 — Section 15d markdown + code cells inserted")
else:
    print("SKIP PATCH 4 (Section 15d already exists)")

# Also need evaluate() to track per_episode_density — patch Section 13
# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 5 — evaluate(): add per_episode_density to returned dict
# Cell 444b4113
# ═══════════════════════════════════════════════════════════════════════════════
eval_cell = by_id("444b4113")

OLD_EVAL_RET = """\
    accs_np  = np.array(accs)
    mean_acc = accs_np.mean()
    ci_95    = 1.96 * accs_np.std() / (len(accs_np) ** 0.5)

    return {
        "mean_acc"    : mean_acc,
        "ci_95"       : ci_95,
        "per_episode" : accs,
        "avg_density" : np.mean(densities),
        "avg_degree"  : np.mean(avg_degrees),
    }"""

NEW_EVAL_RET = """\
    accs_np  = np.array(accs)
    mean_acc = accs_np.mean()
    ci_95    = 1.96 * accs_np.std() / (len(accs_np) ** 0.5)

    return {
        "mean_acc"             : mean_acc,
        "ci_95"                : ci_95,
        "per_episode"          : accs,
        "avg_density"          : np.mean(densities),
        "avg_degree"           : np.mean(avg_degrees),
        "per_episode_density"  : densities,   # Group D: topology_stability per seed
        "avg_intra_ratio"      : np.mean(intra_ratios) if intra_ratios else 0.0,
    }"""

if OLD_EVAL_RET in src(eval_cell):
    set_src(eval_cell, src(eval_cell).replace(OLD_EVAL_RET, NEW_EVAL_RET))
    patches.append("PATCH 5 — evaluate() returns per_episode_density + avg_intra_ratio")
else:
    print("SKIP PATCH 5 (already applied)")

# evaluate() also needs to collect intra_ratios — patch the loop
OLD_EVAL_LOOP = """\
    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            s_imgs, s_lbl, q_imgs, q_lbl = episode_fn(n_way, k_shot, n_query)
            s_imgs = s_imgs.to(DEVICE)
            s_lbl  = s_lbl.to(DEVICE)
            q_imgs = q_imgs.to(DEVICE)
            q_lbl  = q_lbl.to(DEVICE)

            logits, _, mets = model(s_imgs, s_lbl, q_imgs)
            preds = logits.argmax(dim=1)
            acc   = (preds == q_lbl).float().mean().item()
            accs.append(acc)
            densities.append(mets["graph_density"])
            avg_degrees.append(mets["avg_degree"])"""

NEW_EVAL_LOOP = """\
    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            s_imgs, s_lbl, q_imgs, q_lbl = episode_fn(n_way, k_shot, n_query)
            s_imgs = s_imgs.to(DEVICE)
            s_lbl  = s_lbl.to(DEVICE)
            q_imgs = q_imgs.to(DEVICE)
            q_lbl  = q_lbl.to(DEVICE)

            logits, _, mets = model(s_imgs, s_lbl, q_imgs)
            preds = logits.argmax(dim=1)
            acc   = (preds == q_lbl).float().mean().item()
            accs.append(acc)
            densities.append(mets["graph_density"])
            avg_degrees.append(mets["avg_degree"])
            intra_ratios.append(mets.get("intra_edge_ratio", 0.0))"""

if OLD_EVAL_LOOP in src(eval_cell):
    set_src(eval_cell, src(eval_cell).replace(OLD_EVAL_LOOP, NEW_EVAL_LOOP))
    patches.append("PATCH 5b — evaluate() loop collects intra_ratios")
else:
    print("SKIP PATCH 5b (already applied)")

# Add intra_ratios list initialisation
OLD_EVAL_INIT = """\
    model.eval()
    accs, densities, avg_degrees = [], [], []
    intra_ratios, inter_ratios   = [], []"""

NEW_EVAL_INIT = """\
    model.eval()
    accs, densities, avg_degrees = [], [], []
    intra_ratios, inter_ratios   = [], []
    # intra_ratios tracked explicitly for Group D avg_intra_ratio reporting"""

if OLD_EVAL_INIT in src(eval_cell):
    set_src(eval_cell, src(eval_cell).replace(OLD_EVAL_INIT, NEW_EVAL_INIT))
    patches.append("PATCH 5c — evaluate() init comment added")
else:
    # already has the init, skip
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════════════════
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"Patches applied ({len(patches)}):")
for p in patches:
    print(f"  ✓ {p}")
print("=" * 60)
print(f"\nNotebook saved: {NB_PATH}")
