"""
Patch dekae_colab.ipynb to implement Group C (edge feature comparison) ablation.
Modifications:
  1. EdgeIncidenceModule  — add edge_proj_type param ('mlp'|'linear'|'none')
  2. DEKAEModel            — thread edge_proj_type through to each EdgeIncidenceModule
  3. build_model()         — read edge_proj_type from cfg
  4. ABLATION_CONFIGS      — add C1/C2/C3 entries
  5. Section 15b           — add Group C runner block
  6. TOC (cell 1)          — add Group C row
"""

import json, re, sys
from pathlib import Path

NB_PATH = Path(__file__).parent / "dekae_colab.ipynb"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

# ── helper: find cell by id ───────────────────────────────────────────────────
def by_id(cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    cell["source"] = [line + ("\n" if not line.endswith("\n") else "")
                      for line in text.splitlines()]
    # last line should not have trailing newline (notebook convention)
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

patches_applied = []

# ── Cell ID map (raw notebook IDs, no VSC- prefix) ───────────────────────────
# 0  4e915e4e  TOC markdown
# 12 c526db16  EdgeIncidenceModule (Section 6)
# 22 b9806c81  DEKAEModel (Section 11)
# 27 88ed9b71  build_model + train() (Section 12)
# 36 1f9b5c4b  ABLATION_CONFIGS (Section 15)
# 37 a48be1a2  Section 15b execution cell

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 1 — TOC cell: add Group C entry
# ════════════════════════════════════════════════════════════════════════════════
toc_cell = by_id("4e915e4e")
OLD_TOC = "| **15b** | **Execute Ablation Groups A + E ← RUN AFTER 12b** |"
NEW_TOC = ("| **15b** | **Execute Ablation Groups A + E ← RUN AFTER 12b** |\n"
           "| 15c | Ablation Group C — Edge Feature Comparison |")
if OLD_TOC in src(toc_cell):
    set_src(toc_cell, src(toc_cell).replace(OLD_TOC, NEW_TOC))
    patches_applied.append("PATCH 1 — TOC updated")
else:
    print("SKIP PATCH 1 (already applied or marker not found)")

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 2 — EdgeIncidenceModule: add edge_proj_type ('mlp'|'linear'|'none')
# Cell #VSC-b1704a38
# ════════════════════════════════════════════════════════════════════════════════
eim_cell = by_id("c526db16")

OLD_EIM = '''\
class EdgeIncidenceModule(nn.Module):
    """
    Builds the edge incidence matrix B from a given adjacency A,
    then computes edge features E = B^T X W using a shallow MLP.

    Parameters
    ----------
    in_dim  : node feature dimension d
    edge_dim: output edge feature dimension (default = in_dim for skip-compat.)
    hidden  : hidden dim of the 2-layer edge MLP (≤ 64 to avoid overfit)
    """

    def __init__(self, in_dim: int, edge_dim: int = None, hidden: int = 64):
        super().__init__()
        edge_dim = edge_dim or in_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, edge_dim, bias=True),
        )'''

NEW_EIM = '''\
class EdgeIncidenceModule(nn.Module):
    """
    Builds the edge incidence matrix B from a given adjacency A,
    then computes edge features E = B^T X W.

    Parameters
    ----------
    in_dim        : node feature dimension d
    edge_dim      : output edge feature dimension (default = in_dim)
    hidden        : hidden dim of the 2-layer edge MLP (≤ 64 to avoid overfit)
    edge_proj_type: 'mlp'    — 2-layer MLP (default, plan C3)
                    'linear' — single linear layer (plan C2)
                    'none'   — no learned projection; edge feature = zeros
                               (isolates adjacency-only contribution, plan C1)
    """

    def __init__(self, in_dim: int, edge_dim: int = None, hidden: int = 64,
                 edge_proj_type: str = "mlp"):
        super().__init__()
        edge_dim = edge_dim or in_dim
        self.edge_dim       = edge_dim
        self.edge_proj_type = edge_proj_type

        if edge_proj_type == "mlp":
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * in_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, edge_dim, bias=True),
            )
        elif edge_proj_type == "linear":
            # Single linear projection — C2 ablation
            self.edge_mlp = nn.Linear(2 * in_dim, edge_dim, bias=True)
        elif edge_proj_type == "none":
            # No projection — edge features are zeros (C1 ablation)
            # DynamicTopologyModule falls back to bilinear node scores only;
            # EdgeToNodeProjection receives zero vectors → effectively disabled.
            self.edge_mlp = None
        else:
            raise ValueError(f"Unknown edge_proj_type: {edge_proj_type!r}. "
                             "Choose from 'mlp', 'linear', 'none'.")'''

if OLD_EIM in src(eim_cell):
    set_src(eim_cell, src(eim_cell).replace(OLD_EIM, NEW_EIM))
    patches_applied.append("PATCH 2 — EdgeIncidenceModule extended with edge_proj_type")
else:
    print("SKIP PATCH 2 (already applied or marker not found)")

# Also patch the forward method of EdgeIncidenceModule
OLD_EIM_FWD = '''\
    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """
        X : (N, d)  node features
        A : (N, N)  adjacency (may be weighted)

        Returns
        -------
        E         : (|E|, edge_dim)  vector edge features
        src, dst  : edge index tensors
        """
        src, dst = self.adjacency_to_edge_index(A)
        # Concatenate incident node features for each edge
        edge_input = torch.cat([X[src], X[dst]], dim=-1)    # (|E|, 2d)
        E = self.edge_mlp(edge_input)                        # (|E|, edge_dim)
        return E, src, dst'''

NEW_EIM_FWD = '''\
    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """
        X : (N, d)  node features
        A : (N, N)  adjacency (may be weighted)

        Returns
        -------
        E         : (|E|, edge_dim)  vector edge features
                    (zero tensor when edge_proj_type='none')
        src, dst  : edge index tensors
        """
        src, dst = self.adjacency_to_edge_index(A)

        if self.edge_proj_type == "none":
            # C1 ablation: no learned edge features — return zeros.
            # DynamicTopologyModule will rely solely on bilinear node scores.
            # EdgeToNodeProjection receives zero vectors and contributes nothing;
            # the model degenerates to plain message passing over A'.
            E = torch.zeros(len(src), self.edge_dim,
                            device=X.device, dtype=X.dtype)
        else:
            # Concatenate incident node features for each edge
            edge_input = torch.cat([X[src], X[dst]], dim=-1)    # (|E|, 2d)
            E = self.edge_mlp(edge_input)                        # (|E|, edge_dim)
        return E, src, dst'''

if OLD_EIM_FWD in src(eim_cell):
    set_src(eim_cell, src(eim_cell).replace(OLD_EIM_FWD, NEW_EIM_FWD))
    patches_applied.append("PATCH 2b — EdgeIncidenceModule.forward updated")
else:
    print("SKIP PATCH 2b (already applied or marker not found)")

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 3 — DEKAEModel.__init__: add edge_proj_type param
# Cell #VSC-71e6fdfd
# ════════════════════════════════════════════════════════════════════════════════
model_cell = by_id("b9806c81")

OLD_DEKAE_INIT = '''\
    def __init__(self, embed_dim: int = 128, n_gnn_layers: int = 3,
                 rank: int = 16, sparsity_mode: str = "l1",
                 lambda_sparse: float = 0.01, lambda_edge: float = 0.5,
                 topk_k: int = 5, n_way: int = 5, use_dynamic: bool = True,
                 use_edge_proj: bool = True, use_case2: bool = True,
                 edge_dropout: float = 0.1,
                 use_lmt: bool = True, n_mediators: int = 8,
                 lmt_layers: int = 3, lmt_heads: int = 4, lmt_dropout: float = 0.1,
                 lmt_init_strategy: str = "learned", lmt_phases: str = "both",
                 edge_loss_mode: str = "all",
                 adj_residual: float = 0.15):'''

NEW_DEKAE_INIT = '''\
    def __init__(self, embed_dim: int = 128, n_gnn_layers: int = 3,
                 rank: int = 16, sparsity_mode: str = "l1",
                 lambda_sparse: float = 0.01, lambda_edge: float = 0.5,
                 topk_k: int = 5, n_way: int = 5, use_dynamic: bool = True,
                 use_edge_proj: bool = True, use_case2: bool = True,
                 edge_dropout: float = 0.1,
                 use_lmt: bool = True, n_mediators: int = 8,
                 lmt_layers: int = 3, lmt_heads: int = 4, lmt_dropout: float = 0.1,
                 lmt_init_strategy: str = "learned", lmt_phases: str = "both",
                 edge_loss_mode: str = "all",
                 adj_residual: float = 0.15,
                 edge_proj_type: str = "mlp"):
                 # edge_proj_type: 'mlp' (default/C3) | 'linear' (C2) | 'none' (C1)
                 # Controls how EdgeIncidenceModule computes edge features.
                 # Group C ablation (plan §5.3) sweeps these three values.'''

if OLD_DEKAE_INIT in src(model_cell):
    set_src(model_cell, src(model_cell).replace(OLD_DEKAE_INIT, NEW_DEKAE_INIT))
    patches_applied.append("PATCH 3 — DEKAEModel.__init__ signature extended")
else:
    print("SKIP PATCH 3 (already applied or marker not found)")

# Patch the edge_proj_type storage and the EdgeIncidenceModule construction
OLD_STORE = '''\
        self.embed_dim        = embed_dim
        self.n_gnn_layers     = n_gnn_layers
        self.n_way            = n_way
        self.use_dynamic      = use_dynamic
        self.use_edge_proj    = use_edge_proj
        self.use_case2        = use_case2
        self.use_lmt          = use_lmt
        self.edge_loss_mode   = edge_loss_mode
        self.adj_residual     = adj_residual   # anti-collapse: blend A' with k-NN
        self.knn_k            = topk_k
        self.lambda_edge      = lambda_edge'''

NEW_STORE = '''\
        self.embed_dim        = embed_dim
        self.n_gnn_layers     = n_gnn_layers
        self.n_way            = n_way
        self.use_dynamic      = use_dynamic
        self.use_edge_proj    = use_edge_proj
        self.use_case2        = use_case2
        self.use_lmt          = use_lmt
        self.edge_loss_mode   = edge_loss_mode
        self.adj_residual     = adj_residual   # anti-collapse: blend A' with k-NN
        self.knn_k            = topk_k
        self.lambda_edge      = lambda_edge
        self.edge_proj_type   = edge_proj_type  # Group C ablation flag'''

if OLD_STORE in src(model_cell):
    set_src(model_cell, src(model_cell).replace(OLD_STORE, NEW_STORE))
    patches_applied.append("PATCH 3b — DEKAEModel stores edge_proj_type")
else:
    print("SKIP PATCH 3b (already applied or marker not found)")

# Patch EdgeIncidenceModule instantiation inside DEKAEModel to pass edge_proj_type
OLD_EIM_INST = '''\
        # Edge incidence modules (shared across both cases)
        self.edge_modules = nn.ModuleList([
            EdgeIncidenceModule(embed_dim, embed_dim, hidden=64)
            for _ in range(n_gnn_layers)
        ])'''

NEW_EIM_INST = '''\
        # Edge incidence modules (shared across both cases)
        # edge_proj_type controls Group C ablation: 'mlp'(C3) / 'linear'(C2) / 'none'(C1)
        self.edge_modules = nn.ModuleList([
            EdgeIncidenceModule(embed_dim, embed_dim, hidden=64,
                                edge_proj_type=edge_proj_type)
            for _ in range(n_gnn_layers)
        ])'''

if OLD_EIM_INST in src(model_cell):
    set_src(model_cell, src(model_cell).replace(OLD_EIM_INST, NEW_EIM_INST))
    patches_applied.append("PATCH 3c — DEKAEModel passes edge_proj_type to EdgeIncidenceModules")
else:
    print("SKIP PATCH 3c (already applied or marker not found)")

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 4 — build_model(): read edge_proj_type from cfg
# Cell #VSC-f544ca03
# ════════════════════════════════════════════════════════════════════════════════
train_cell = by_id("88ed9b71")

OLD_BUILD = '''\
    return DEKAEModel(
        embed_dim     = cfg["embed_dim"],
        n_gnn_layers  = cfg["n_gnn_layers"],
        rank          = cfg["rank"],
        sparsity_mode = cfg["sparsity_mode"],
        lambda_sparse = cfg["lambda_sparse"],
        lambda_edge   = cfg["lambda_edge"],
        topk_k        = cfg["topk_k"],
        n_way         = cfg["n_way"],
        use_edge_proj      = cfg.get("use_edge_proj", True),
        use_case2          = cfg.get("use_case2", True),         # plan §5.1: A4(False) vs A5(True)
        use_lmt            = cfg.get("use_lmt", True),           # plan §5.8: Group H ablation
        n_mediators        = cfg.get("n_mediators", 8),
        lmt_layers         = cfg.get("lmt_layers", 3),
        lmt_heads          = cfg.get("lmt_heads", 4),
        lmt_init_strategy  = cfg.get("lmt_init_strategy", "learned"),  # H4 ablation
        lmt_phases         = cfg.get("lmt_phases", "both"),             # H5 ablation
        edge_loss_mode     = cfg.get("edge_loss_mode", "all"),          # Group I ablation
    ).to(DEVICE)'''

NEW_BUILD = '''\
    return DEKAEModel(
        embed_dim     = cfg["embed_dim"],
        n_gnn_layers  = cfg["n_gnn_layers"],
        rank          = cfg["rank"],
        sparsity_mode = cfg["sparsity_mode"],
        lambda_sparse = cfg["lambda_sparse"],
        lambda_edge   = cfg["lambda_edge"],
        topk_k        = cfg["topk_k"],
        n_way         = cfg["n_way"],
        use_edge_proj      = cfg.get("use_edge_proj", True),
        use_case2          = cfg.get("use_case2", True),         # plan §5.1: A4(False) vs A5(True)
        use_lmt            = cfg.get("use_lmt", True),           # plan §5.8: Group H ablation
        n_mediators        = cfg.get("n_mediators", 8),
        lmt_layers         = cfg.get("lmt_layers", 3),
        lmt_heads          = cfg.get("lmt_heads", 4),
        lmt_init_strategy  = cfg.get("lmt_init_strategy", "learned"),  # H4 ablation
        lmt_phases         = cfg.get("lmt_phases", "both"),             # H5 ablation
        edge_loss_mode     = cfg.get("edge_loss_mode", "all"),          # Group I ablation
        edge_proj_type     = cfg.get("edge_proj_type", "mlp"),          # Group C ablation
    ).to(DEVICE)'''

if OLD_BUILD in src(train_cell):
    set_src(train_cell, src(train_cell).replace(OLD_BUILD, NEW_BUILD))
    patches_applied.append("PATCH 4 — build_model() reads edge_proj_type from cfg")
else:
    print("SKIP PATCH 4 (already applied or marker not found)")

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 5 — ABLATION_CONFIGS: add C1, C2, C3 entries
# Cell #VSC-a66024c3
# ════════════════════════════════════════════════════════════════════════════════
abl_cell = by_id("1f9b5c4b")

# Insert Group C block right after Group B block
OLD_GROUP_B_END = '''\
    "B3_dynamic":   {**BASE, "use_dynamic": True,  "use_lmt": False,
                     "lambda_edge": 0.5},

    # ── Group E: Sparsity ─────────────────────────────────────────────────────'''

NEW_GROUP_B_C = '''\
    "B3_dynamic":   {**BASE, "use_dynamic": True,  "use_lmt": False,
                     "lambda_edge": 0.5},

    # ── Group C: Edge Feature Projection Type (plan §5.3) ────────────────────
    # Goal: isolate the contribution of learned edge features vs. raw adjacency.
    #
    # C1 — No edge features: EdgeIncidenceModule returns zeros.
    #       DynamicTopologyModule uses bilinear node scores only.
    #       EdgeToNodeProjection receives zero vectors → degenerates to A'H MP.
    #       This is the "just adjacency" baseline from the plan.
    #
    # C2 — Linear projection: E = W·[h_i||h_j], single linear layer, no ReLU.
    #       Tests whether a non-linear MLP is needed, or if linear suffices.
    #
    # C3 — MLP projection: 2-layer MLP with ReLU (current default architecture).
    #       Should outperform C1 and C2 — validates that non-linear edge features
    #       are worth the extra parameters.
    #
    # All C variants use: use_case2=True, use_dynamic=True, use_lmt=False
    # (LMT is off so the GNN edge-feature contribution is cleanly isolated).
    # The default lambda_edge=0.5 is kept so the edge correction loss engages.
    #
    # Expected ordering: C3 (MLP) ≥ C2 (linear) > C1 (none)
    # If C2 ≈ C3: linear projection suffices, MLP doesn't add value (report this)
    # If C1 ≈ C2: learned projection itself is not the bottleneck (review arch)
    "C1_no_edge_feat":  {**BASE,
        "use_dynamic": True, "use_edge_proj": True,
        "use_case2": True,   "use_lmt": False,
        "lambda_edge": 0.5,  "edge_proj_type": "none"},

    "C2_linear_proj":   {**BASE,
        "use_dynamic": True, "use_edge_proj": True,
        "use_case2": True,   "use_lmt": False,
        "lambda_edge": 0.5,  "edge_proj_type": "linear"},

    "C3_mlp_proj":      {**BASE,
        "use_dynamic": True, "use_edge_proj": True,
        "use_case2": True,   "use_lmt": False,
        "lambda_edge": 0.5,  "edge_proj_type": "mlp"},    # same as A4_case2 — sanity anchor

    # ── Group E: Sparsity ─────────────────────────────────────────────────────'''

if OLD_GROUP_B_END in src(abl_cell):
    set_src(abl_cell, src(abl_cell).replace(OLD_GROUP_B_END, NEW_GROUP_B_C))
    patches_applied.append("PATCH 5 — ABLATION_CONFIGS: C1/C2/C3 added")
else:
    print("SKIP PATCH 5 (already applied or marker not found)")

# Also update the print statements at the bottom of the ablation cell
OLD_ABL_PRINT = '''\
print("Ablation configs defined.")
print("Group A variants :", [k for k in ABLATION_CONFIGS if k.startswith("A")])
print("Group E variants :", [k for k in ABLATION_CONFIGS if k.startswith("E")])'''

NEW_ABL_PRINT = '''\
print("Ablation configs defined.")
print("Group A variants :", [k for k in ABLATION_CONFIGS if k.startswith("A")])
print("Group C variants :", [k for k in ABLATION_CONFIGS if k.startswith("C")])
print("Group E variants :", [k for k in ABLATION_CONFIGS if k.startswith("E")])'''

if OLD_ABL_PRINT in src(abl_cell):
    set_src(abl_cell, src(abl_cell).replace(OLD_ABL_PRINT, NEW_ABL_PRINT))
    patches_applied.append("PATCH 5b — ABLATION_CONFIGS print statement updated")
else:
    print("SKIP PATCH 5b (already applied or marker not found)")

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 6 — Section 15b execution cell: add Group C runner
# Cell #VSC-d0777c12
# ════════════════════════════════════════════════════════════════════════════════
exec_cell = by_id("a48be1a2")

OLD_SAVE_BLOCK = '''\
# ── Save all ablation results to Drive ────────────────────────────────────────
all_ablation = group_a_results + group_e_core_results
with open(str(RESULTS_DIR / "ablation_results.json"), "w") as f:
    import json
    save = [{k: v for k, v in r.items() if k != "per_episode"} for r in all_ablation]
    json.dump(save, f, indent=2)'''

NEW_SAVE_BLOCK = '''\
# ── Group C: Edge Feature Projection Type (plan §5.3) ────────────────────────
# Tests whether learned edge features are necessary vs. linear or no projection.
# Estimated time per variant on T4: ~60–90 min (same as Group A variants).
# Run after Groups A & E to know the baseline accuracy for comparison.
#
# Expected result: C3_mlp_proj ≥ C2_linear_proj > C1_no_edge_feat
# C3 should match A4_case2 (same config) — acts as a sanity anchor.
RUN_GROUP_C = False   # ← set True after Groups A & E complete

if RUN_GROUP_C:
    print("\nRunning Group C ablations (edge feature projection type)…")
    group_c_results = run_all_ablations(
        groups            = ["C1_no_edge_feat", "C2_linear_proj", "C3_mlp_proj"],
        episode_fn_train  = train_sampler_aug,
        episode_fn_val    = val_sampler,
    )
    print_ablation_table(group_c_results, "Group C — Edge Feature Projection Type")

    # ── Parameter count comparison ────────────────────────────────────────────
    # Small parameter delta between C1/C2/C3 proves the gain is from
    # representation quality, not extra capacity.
    print("\n── Parameter counts per Group C variant ──")
    for name in ["C1_no_edge_feat", "C2_linear_proj", "C3_mlp_proj"]:
        m = build_model(ABLATION_CONFIGS[name])
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:<22}: {n:>10,} params")

    # ── Graph density check per variant ───────────────────────────────────────
    # C1 (no edge features) may produce denser graphs because the edge scorer
    # inside DynamicTopologyModule has no edge-feature signal — it relies solely
    # on bilinear node scores which tend to be less discriminative.
    # If density(C1) >> density(C3), it suggests edge features are essential
    # for learning meaningful sparse topology.
    print("\n── Graph density check (C1 vs C3, single episode) ──")
    _s_c = torch.randn(5, 3, 84, 84).to(DEVICE)
    _l_c = torch.arange(5).to(DEVICE)
    _q_c = torch.randn(15, 3, 84, 84).to(DEVICE)
    for name in ["C1_no_edge_feat", "C3_mlp_proj"]:
        _mc = build_model(ABLATION_CONFIGS[name])
        _mc.eval()
        with torch.no_grad():
            _, _, _met_c = _mc(_s_c, _l_c, _q_c)
        print(f"  {name:<22}: density={_met_c['graph_density']:.3f}  "
              f"avg_degree={_met_c['avg_degree']:.2f}")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(str(RESULTS_DIR / "ablation_group_c.json"), "w") as _f:
        import json as _json
        _save_c = [{k: v for k, v in r.items() if k != "per_episode"}
                   for r in group_c_results]
        _json.dump(_save_c, _f, indent=2)
    print(f"\nGroup C results saved → {RESULTS_DIR}/ablation_group_c.json")
else:
    print("  (Group C skipped — set RUN_GROUP_C=True after Groups A & E complete)")
    group_c_results = []

# ── Save all ablation results to Drive ────────────────────────────────────────
all_ablation = group_a_results + group_e_core_results + group_c_results
with open(str(RESULTS_DIR / "ablation_results.json"), "w") as f:
    import json
    save = [{k: v for k, v in r.items() if k != "per_episode"} for r in all_ablation]
    json.dump(save, f, indent=2)'''

if OLD_SAVE_BLOCK in src(exec_cell):
    set_src(exec_cell, src(exec_cell).replace(OLD_SAVE_BLOCK, NEW_SAVE_BLOCK))
    patches_applied.append("PATCH 6 — Section 15b: Group C runner added")
else:
    print("SKIP PATCH 6 (already applied or marker not found)")

# ════════════════════════════════════════════════════════════════════════════════
# PATCH 7 — Insert new markdown + code cells for Section 15c
# after cell #VSC-d0777c12 (cell 38, Section 15b)
# ════════════════════════════════════════════════════════════════════════════════
# Find index of exec_cell in cells list
exec_idx = next(i for i, c in enumerate(cells) if c.get("id") == "a48be1a2")

# Check if Section 15c already exists
already_has_15c = any(c.get("id") in ("grpc0001", "grpc0002") for c in cells)

if not already_has_15c:
    md_15c = {
        "cell_type": "markdown",
        "id": "grpc0001",
        "metadata": {},
        "source": [
            "## Section 15c: Ablation Group C — Edge Feature Projection Type\n",
            "\n",
            "Compares three levels of edge feature richness (plan §5.3):\n",
            "\n",
            "| Variant | Edge Projection | Description |\n",
            "|---------|----------------|-------------|\n",
            "| **C1** | `none` | No learned edge features — zeros passed to `DynamicTopologyModule` and `EdgeToNodeProjection`; degenerates to adjacency-only MP |\n",
            "| **C2** | `linear` | Single `nn.Linear(2d → d)` — low-capacity linear projection, no non-linearity |\n",
            "| **C3** | `mlp` | 2-layer MLP with ReLU (current default) — full expressive edge representation |\n",
            "\n",
            "**What this proves**: The gain from DEKAE's edge-feature design comes from\n",
            "non-linear relational representation, not just from having more parameters.\n",
            "If C3 > C2 > C1, the full MLP is justified. If C2 ≈ C3, a linear projection\n",
            "suffices and the MLP can be simplified without accuracy loss.\n",
            "\n",
            "All Group C variants use `use_case2=True`, `use_dynamic=True`, `use_lmt=False`\n",
            "so the edge-feature contribution to the GNN is cleanly isolated.\n",
            "The `edge_proj_type` flag added to `EdgeIncidenceModule` and `DEKAEModel`\n",
            "controls this ablation; `build_model()` reads it from the config dict."
        ],
        "outputs": [],
        "execution_count": None
    }

    code_15c = {
        "cell_type": "code",
        "id": "grpc0002",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# ── Section 15c: Group C — Edge Feature Projection Type ──────────────────────\n",
            "# Run this cell independently to execute only Group C ablations.\n",
            "# Requires a trained full model (Section 12b) so relative performance is\n",
            "# measured against a meaningful representation, not random features.\n",
            "#\n",
            "# Each variant uses the BASE ablation config (100 epochs × 300 ep/epoch).\n",
            "# Estimated time: 3 variants × ~60–90 min ≈ 3–4.5 hours on T4.\n",
            "#\n",
            "# ── Quick Model Sanity Check (runs immediately, no training needed) ─────────\n",
            "print('Group C model sanity checks (untrained):')\n",
            "print('-' * 55)\n",
            "_s_c = torch.randn(5, 3, 84, 84).to(DEVICE)\n",
            "_l_c = torch.arange(5).to(DEVICE)\n",
            "_q_c = torch.randn(15, 3, 84, 84).to(DEVICE)\n",
            "\n",
            "for _name_c, _ept in [('C1_no_edge_feat', 'none'),\n",
            "                       ('C2_linear_proj',  'linear'),\n",
            "                       ('C3_mlp_proj',     'mlp')]:\n",
            "    _mc = build_model({**BASE, 'edge_proj_type': _ept,\n",
            "                       'use_dynamic': True, 'use_edge_proj': True,\n",
            "                       'use_case2': True, 'use_lmt': False,\n",
            "                       'lambda_edge': 0.5})\n",
            "    _mc.eval()\n",
            "    with torch.no_grad():\n",
            "        _lg_c, _al_c, _mt_c = _mc(_s_c, _l_c, _q_c)\n",
            "    _np_c = sum(p.numel() for p in _mc.parameters())\n",
            "    print(f'  {_name_c:<22} edge_proj={_ept:<7}  '\n",
            "          f'logits={list(_lg_c.shape)}  '\n",
            "          f'density={_mt_c[\"graph_density\"]:.3f}  '\n",
            "          f'params={_np_c:,}')\n",
            "\n",
            "print()\n",
            "print('All three variants produce valid logits. '\n",
            "      'Set RUN_GROUP_C=True in Section 15b to run full training.')\n"
        ]
    }

    # Insert after exec_cell (index exec_idx)
    cells.insert(exec_idx + 1, md_15c)
    cells.insert(exec_idx + 2, code_15c)
    patches_applied.append("PATCH 7 — Section 15c markdown + sanity-check code cells inserted")
else:
    print("SKIP PATCH 7 (Section 15c already exists)")

# ════════════════════════════════════════════════════════════════════════════════
# Write notebook
# ════════════════════════════════════════════════════════════════════════════════
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"Patches applied ({len(patches_applied)}):")
for p in patches_applied:
    print(f"  ✓ {p}")
print("=" * 60)
print(f"\nNotebook saved: {NB_PATH}")
