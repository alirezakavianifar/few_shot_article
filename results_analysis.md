# DEKAE Experiment Results Analysis
**Dataset**: CIFAR-FS | **Protocol**: 5-way episodic | **Date**: February 2026

---

## 1. Main Training Results (Full Model, 100 Epochs)

| Metric | Value |
|---|---|
| Training configuration | 100 epochs, 300 episodes/epoch |
| Warm-up | 5 epochs (static graph) |
| Edge-loss ramp | 20 epochs |
| Sparsity mode | L1 |
| λ_edge | 0.5 |
| Best val accuracy | **41.10%** (epoch 100) |
| 5-way 1-shot test accuracy | **45.22% ± 0.90%** (600 episodes) |
| 5-way 5-shot test accuracy | **55.54% ± 0.80%** (600 episodes) |
| Average graph density (test) | 0.080 |
| Average node degree (test) | 6.32 |
| Total parameters | 454,920 |

### Context vs. Expected Performance
The FSAKE paper reports **68.9%** (5-way 1-shot) on miniImageNet. CIFAR-FS is typically **5–8% higher** than miniImageNet for the same method, which implies a competitive method should achieve **~74–77%** on CIFAR-FS. Our full model at **45.22%** is roughly **28–32 percentage points below** that expectation — a critically underperforming result.

### Training Curve
| Epoch | Train Acc | Val Acc | Density |
|---|---|---|---|
| 5 | 0.3158 | 0.3163 | 0.086 |
| 25 | 0.3532 | 0.3624 | 0.026 |
| 50 | 0.3845 | 0.3727 | 0.076 |
| 65 | 0.3822 | **0.3901** | 0.084 |
| 85 | 0.4082 | 0.3981 | 0.086 |
| 100 | 0.3991 | **0.4110** | 0.079 |

**Observation**: The model converges very slowly and plateaus in the 39–41% range after epoch 65. There is no sign of continued improvement. Training and validation accuracy stay close throughout, suggesting underfitting rather than overfitting.

---

## 2. Group A Ablation — Topology Dynamics

All variants were trained for **30 epochs, 100 episodes/epoch**, evaluated on 100 test episodes.

| Variant | Description | Accuracy | CI95 | Density |
|---|---|---|---|---|
| A1_FSAKE | Static k-NN (baseline) | **35.25%** | ±1.90% | 0.086 |
| A2_dyn_only | Dynamic rewiring only | 35.16% | ±1.82% | 0.059 |
| A3_dyn_proj | Dynamic + edge→node projection | 34.89% | ±1.80% | 0.025 |
| A4_DEKAE_full | Dynamic + projection + edge loss | 34.52% | ±1.92% | 0.064 |

### Key Observations

**Each novelty marginally degrades performance**. The full DEKAE model (A4) scores 0.73pp *below* the FSAKE static baseline. All differences are within the 95% CI, so none are statistically significant — but the consistent downward trend across A1→A4 is a concerning pattern.

- `A2 − A1 = −0.09pp`: Dynamic rewiring alone is indistinguishable from static k-NN, or slightly harmful.
- `A3 − A2 = −0.27pp`: The edge-to-node projection add no measurable benefit at 30 epochs.
- `A4 − A3 = −0.37pp`: The edge correction loss has a small negative effect in short training, possibly because the backbone is not yet expressive enough at this stage for the edge supervision signal to be meaningful.

**Density decreases as novelties are added**: A1 stays at 0.086 (close to the initial k-NN density), while A3 collapses to 0.025 — an extremely sparse graph. This over-sparsification from the L1 penalty combined with the edge projection is likely degrading message passing quality.

---

## 3. Group E Ablation — Sparsity Regularisation

| Variant | Sparsity Method | Accuracy | CI95 | Density |
|---|---|---|---|---|
| E2_l1 | L1 on A' | **34.68%** | ±1.88% | 0.068 |
| E1_no_sparse | None | 34.63% | ±1.81% | 0.037 |
| E3_topk | Top-k hard mask | 34.57% | ±1.80% | 0.063 |
| E4_laplacian | Laplacian smoothness | 26.24% | ±1.55% | 0.047 |

### Key Observations

- **L1, no-sparsity, and top-k are statistically indistinguishable** (~34.6–34.7%). All three within each other's CI. The choice of sparsity mechanism has negligible impact at this training scale.
- **E1 (no sparsity) still converges to a sparse graph** (density 0.037) — the model is self-regularising through other mechanisms. This means the L1 penalty is redundant as currently tuned.
- **E4 (Laplacian) is catastrophically harmful**: accuracy collapsed to **26.24%**, only marginally above random chance (20%). The Laplacian smoothness term $\text{tr}(X'^T L X')$ over-smoothes node representations in the few-shot regime, effectively erasing class-discriminative information. The term likely conflicts with the classification loss at the small episode scale (5-way, 80 nodes).
- The plan's prediction that "E1 demonstrates the collapse baseline" was **not confirmed** — no graph collapse occurred without regularisation, and density was even lower in E1 than E2/E3.

---

## 4. Synthetic Topology Recovery Experiment

The experiment was run on an **untrained model** (fresh random initialisation).

| Noise σ | k-NN F1 | DEKAE F1 |
|---|---|---|
| 0.3 | 0.335 ± 0.040 | **0.013 ± 0.017** |
| 0.6 | 0.233 ± 0.039 | 0.017 ± 0.017 |
| 0.8 | 0.221 ± 0.038 | 0.014 ± 0.018 |
| 1.0 | 0.230 ± 0.038 | 0.012 ± 0.020 |

### Critical Observation

The untrained DEKAE model scores **near-zero F1** (~0.01–0.02) across all noise levels, while even the simple k-NN baseline achieves **0.22–0.34 F1**. This is expected since the model was not trained, but the experiment as run does **not** evaluate the trained model's topology recovery quality. The synthetic experiment must be re-run using the best checkpoint to produce a meaningful result. As presented, this data cannot support any claim about the model's structural recovery capability.

---

## 5. Graph Structure Observations

### Density Behaviour
- Initial (sanity check, random init): density = **0.358** at N=20 nodes
- After training (full model): stable at **~0.079–0.086**, avg degree ~6.32
- Density fluctuates between 0.025 and 0.086 across training epochs — not monotonically stable
- Over-sparsification observed in A3_dyn_proj (density 0.025), likely impairing message passing

### Degree Distribution (from visualisation)
- Most nodes cluster around degree 5–7
- Long right tail reaching degree 17 for a small subset of nodes
- This confirms some variable-degree behaviour is occurring: certain nodes accumulate significantly more edges
- The distribution is **right-skewed, not uniform**, which is a positive signal for the variable-degree hypothesis

### Edge Topology (Episode Topology Plot)
- The visualised graph shows **cross-class edges dominate** — edges frequently connect nodes of different colours (classes)
- Intra-class clustering is not visually apparent in the learned topology
- This is consistent with the near-zero F1 in topology recovery: the model has not learned to preferentially connect same-class nodes

---

## 6. Summary of Novelties Performance

| Novelty | Expected Benefit (per plan.md) | Observed Outcome |
|---|---|---|
| Dynamic topology rewiring | Replace static k-NN with learned adjacency | Neutral to slightly negative vs. static k-NN |
| Edge incidence formulation ($B$, $E=BXW$) | Explicit vector edge features | No measurable accuracy gain in ablation |
| Edge-to-node projection ($X' = A'EW'$) | Higher-order relational propagation | Slight degradation (A3 < A2) |
| Edge correction loss (support-only) | Supervised topology recovery | Small negative at short training; full model improves to +10pp over ablation with more training |
| L1 sparsity regularisation | Prevent graph collapse | Effective at maintaining sparse graphs; minimal accuracy impact |
| Warm-up + curriculum edge-loss ramp | Training stability | Training is stable (no gradient explosions observed) |
| Laplacian smoothness | Topology regularisation | **Harmful** — collapsed accuracy to near-random (26.24%) |

---

## 7. Diagnosis — Why Performance Is Below Expectations

### 7.1 Underfitting, Not Overfitting
Train and val accuracy remain closely aligned throughout (e.g., epoch 100: train 0.3991, val 0.4110). The model is not memorising episodes. The fundamental issue is that the model is not learning discriminative features adequately within the Conv4 backbone at this training scale.

### 7.2 Short Ablation Training
The ablation group (30 epochs × 100 episodes = 3,000 episodes) vs. the full run (100 epochs × 300 = 30,000 episodes) shows a large gap (~35% vs ~45%). This means **the ablation table is not a fair comparison** of the methods' true maximum capability — it shows early-training behaviour only. Full training across all ablation variants would be needed for reliable conclusions.

### 7.3 Edge Correction Loss Not Engaging at Initialisation
The sanity check output `Edge correction loss (support-only): 0.0000` at random initialisation suggests the support embeddings may start nearly orthogonal to each other, making the cosine margin loss trivially satisfied before any learning occurs. This could delay the signal from the edge loss impacting the backbone for many epochs.

### 7.4 Synthetic Experiment Run on Untrained Model
The topology recovery experiment must be re-run with the trained checkpoint. The current data is uninformative about whether training improves structural recovery.

### 7.5 Laplacian Regulariser Incompatibility
The Laplacian smoothness term $\mathcal{L}_{smooth} = \text{tr}(X'^T L X')$ forces adjacent nodes toward similar representations, which in a few-shot episode means nodes of different classes that share an edge become more similar — directly harming class separation. In the few-shot setting with mixed-class graphs, this is a particularly destructive inductive bias.

### 7.6 Over-sparsification Under Certain Configurations
Density of 0.025 in A3 suggests the edge projection pathway is causing the model to prune almost all edges, essentially blocking message passing. This may explain A3 being worse than A2.

---

## 8. Next Steps / Recommendations

1. **Re-run ablations with full training (100 epochs, 300 ep/epoch)** to get comparable final performance across A1–A4. The current 30-epoch ablations are insufficient to draw conclusions.
2. **Re-run synthetic experiment with the trained checkpoint** instead of the untrained model.
3. **Remove Laplacian regularisation** from the candidate set — it is empirically harmful in this setting.
4. **Tune the sparsity coefficient** ($\lambda_{sparse}$): the current density (~0.025–0.086) varies widely; stabilise it around 0.10–0.15 to ensure adequate message passing.
5. **Investigate the edge correction loss warm-up more carefully**: log `intra_edge_ratio` and `inter_edge_ratio` over training to confirm the edge loss is actually separating classes in the topology.
6. **Extend training**: the main model is still improving at epoch 100. Try 200–300 epochs with a cosine LR schedule.
7. **Compare against ProtoNet baseline** to understand how much of the performance gap is attributable to the Graph component vs. the backbone + episodic protocol setup.
