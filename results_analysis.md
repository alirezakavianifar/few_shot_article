# DEKAE — Colab Run Results Analysis
**Date**: February 23, 2026  
**Dataset**: CIFAR-FS · 5-way 1-shot & 5-way 5-shot  
**Training**: 200 epochs × 300 episodes, T4 GPU (~75 min)

---

## 1. Summary of Obtained Results

### 1.1 Main Test Evaluation (600 episodes, best checkpoint at epoch 170)

| Method | 5-way 1-shot | 5-way 5-shot | Avg Density | Avg Degree |
|---|---|---|---|---|
| DEKAE (ours) | **50.16 ± 1.05%** | **60.52 ± 0.83%** | 0.081 | 6.38 |

### 1.2 Ablation Group A — Topology Dynamics (100 epochs, 200 eval episodes)

| Variant | Description | Acc (%) | CI95 | Density | Intra-E |
|---|---|---|---|---|---|
| A3_dyn_proj | Dynamic + edge-to-node proj, no edge loss | **44.14%** | ±1.64% | 0.088 | 0.000 |
| A1_FSAKE | Static k-NN (reproduced baseline) | 43.41% | ±1.57% | 0.089 | 0.000 |
| A4_DEKAE_full | Full model (A3 + edge correction loss) | 42.88% | ±1.52% | 0.054 | 0.000 |
| A2_dyn_only | Dynamic topology only, no edge proj | 36.26% | ±1.42% | 0.088 | 0.000 |
| A0_ProtoNet | Pure prototypical classifier | 36.00% | ±1.41% | 0.082 | 0.000 |

### 1.3 Ablation Group E — Sparsity Regularisation (100 epochs, 200 eval episodes)

| Variant | Sparsity | Acc (%) | CI95 | Density |
|---|---|---|---|---|
| E3_topk | Top-k hard mask | **43.55%** | ±1.57% | 0.063 |
| E2_l1 | L1 on A' | 41.88% | ±1.58% | 0.088 |
| E1_no_sparse | None | 41.70% | ±1.54% | 0.089 |

### 1.4 Synthetic Topology Recovery (30 trials per noise level)

| σ | kNN F1 | DEKAE (untrained) F1 | DEKAE (trained) F1 |
|---|---|---|---|
| 0.3 | 0.3382 | 0.0198 | 0.0435 |
| 0.6 | 0.2443 | 0.0140 | 0.0247 |
| 0.8 | 0.2368 | 0.0198 | 0.0325 |
| 1.0 | 0.2217 | 0.0152 | 0.0236 |

---

## 2. Are These Results Worthy of Publication? — **No, Not Yet**

The current results are **not publishable as-is** for the following critical reasons, ranked by severity.

---

## 3. Critical Failures

### 3.1 🔴 DEKAE Full Model Does NOT Beat the Baseline (Fatal)

The paper's core claim is that DEKAE improves over FSAKE. The ablation reveals the opposite:

| | Acc @ 100 epochs |
|---|---|
| FSAKE (A1) | 43.41% |
| DEKAE full (A4) | 42.88% |

DEKAE full is **0.53 percentage points below its own baseline**. The edge correction loss (the primary novel contribution) actively hurts performance relative to simply using dynamic topology with edge-to-node projection (A3 = 44.14%). A reviewer will immediately reject a paper where the proposed method underperforms the baseline it claims to extend.

### 3.2 🔴 Edge Correction Loss Has Zero Effect (Fatal)

The `intra_edge_ratio` is **0.000 across every single epoch**, across every variant, including A4 (the full model with edge loss enabled). This metric measures whether the learned graph places more edges between same-class support nodes than different-class nodes. A value of exactly 0.000 for 200 training epochs and 5 ablation runs means one of three things:

- The edge loss computation branch is silently being skipped (a code path bug).
- The support mask is not being applied correctly, so no support–support edges ever receive gradient from `edge_correction_loss`.
- The loss is numerically zero (e.g., all cosine similarities are below the contrastive margin, making the loss term trivially satisfied at initialisation and never receiving a useful gradient signal).

This must be diagnosed before any further training. Until `intra_edge_ratio` rises clearly above 0.000 at training time, the supervised topology recovery claim has no empirical support.

### 3.3 🔴 Topology Recovery is Worse Than k-NN (Fatal for Section 2.3 Claim)

The plan explicitly requires DEKAE to outperform k-NN on the planted-partition benchmark to validate the structural recovery claim. The actual result is the reverse:

| σ | kNN F1 | DEKAE trained F1 |
|---|---|---|
| 0.3 | **0.338** | 0.044 |
| 1.0 | **0.222** | 0.024 |

DEKAE trained achieves **7–12% of the kNN F1**. This means the dynamic topology is producing meaningless edge structure rather than recovering class-consistent adjacency. In the paper, the synthetic experiment is mandatory; if DEKAE cannot beat a 5-nearest-neighbours heuristic in topology recovery, Contribution 3 in the paper identity statement (§1.6) is unsupported.

### 3.4 🔴 Absolute Accuracies Are Far Below Literature Baselines

Competitive CIFAR-FS 5-way 1-shot baselines from 2020–2023 literature:

| Method | CIFAR-FS 5-way 1-shot |
|---|---|
| FEAT (2020) | ~68% |
| DeepEMD (2020) | ~75% |
| FSAKE (reported, miniImageNet) | 68.9% → expected ~74–77% on CIFAR-FS |
| **DEKAE (reproduced FSAKE, A1)** | **43.41%** |
| **DEKAE full (A4)** | **42.88%** |

The reproduced FSAKE baseline at 43.41% is roughly **30 percentage points below the expected competitive figure**. This indicates a fundamental problem with the implementation (backbone underfitting, normalisation error, or incorrect loss computation) independent of the DEKAE contributions. No reviewer will accept a comparison table built on a 43% "baseline" when the original FSAKE reports 68.9%.

The pure ProtoNet (A0) at 36.00% is also anomalously low — even basic ProtoNets on CIFAR-FS should yield 55–60% after proper training.

---

## 4. Secondary Problems

### 4.1 🟡 Training Plateau / No Convergence in Second Half

Training validation accuracy progression:

| Epoch | Val Acc |
|---|---|
| 50 | 38.57% |
| 100 | 45.33% |
| 115 | **47.99%** (best) |
| 170 | **47.97%** (best re-achieved) |
| 200 | 45.30% |

The model effectively stopped improving after epoch 115. The 85 additional epochs (115–200) added no measurable gain. This suggests either (a) the 200-epoch CosineAnnealingLR schedule is too conservative, causing premature flat LR in the 100–200 range, or (b) the model has genuinely converged to a poor local optimum due to the broken edge-loss geometry.

### 4.2 🟡 Graph Density Instability

Density fluctuates between 0.019 and 0.092 across epochs with no monotonic stabilisation trend. The plan requires a "Graph Density curve demonstrating sparsity stabilization, not collapse." The current density trajectory does not tell a coherent story and would not pass the visualisation requirement in Section 6.5 of the plan.

### 4.3 🟡 Ablation A4 < A3 Breaks the Additive Contribution Narrative

The plan's ablation table (Groups A, §5.1) is designed to show:

```
A2 − A1 → dynamic topology adds value
A3 − A2 → edge-to-node projection adds value
A4 − A3 → edge supervision adds value
```

The actual deltas are:
- A2 − A1 = 36.26 − 43.41 = **−7.15 pp** (dynamic-only hurts)
- A3 − A2 = 44.14 − 36.26 = **+7.88 pp** (edge projection helps)
- A4 − A3 = 42.88 − 44.14 = **−1.26 pp** (edge supervision hurts)

The narrative requires all three deltas to be positive. Two of three are currently negative or wrong-sign.

### 4.4 🟡 E3 (Top-k) Outperforms E2 (L1) — Contradicts Paper Design Choice

The plan explicitly states the final model must use L1 (soft sparsity) to preserve the variable-degree claim, with top-k used only as an ablation baseline. However E3 (top-k) outperforms E2 (L1) by 1.67 pp (43.55% vs 41.88%). This creates a contradiction: the best-performing sparsity mechanism is not the one designated as the "proposed method." A reviewer will ask why the paper does not use top-k as the primary design.

---

## 5. Root Cause Hypotheses (Priority Diagnosis Order)

### H1 — Edge Correction Loss Is Silently Skipped (Highest Priority)

The persistent `intra_edge_ratio = 0.000` strongly suggests the `edge_correction_loss()` function either produces a zero tensor every call or its output is not contributing to the backward pass. Recommended diagnostic:

```python
# Add inside the training loop at each episode:
print(f"edge_loss_raw: {e_loss_l.item():.6f}, lambda_scale: {lambda_edge_scale:.4f}")
print(f"support-support pairs found: {ss_mask.sum().item()}")
```

If `support-support pairs found` is always 0, the support mask is broken. If `edge_loss_raw` is always 0.000000, the cosine similarity targets are trivially satisfied.

### H2 — Backbone Underfitting (High Priority)

ProtoNet at 36% on CIFAR-FS (expected 55–60%) suggests the Conv4 backbone is not learning competitive embeddings. Possible causes: learning rate too low (5e-4 with Adam is reasonable but may need tuning), weight decay too aggressive, or normalisation mismatch between training and the CIFAR-FS statistics. Try:
- Removing BatchNorm `track_running_stats` issues under few-shot episodic training.
- Increasing backbone capacity or checking that the backbone gradient is not being blocked.
- Confirming CIFAR-FS images are loaded at the correct 84×84 resolution and normalised with the correct mean/std.

### H3 — Cosine Classifier Geometry (Medium Priority)

The logits are computed as cosine similarity between query embeddings and prototype embeddings. If embeddings are not L2-normalised for the prototype computation, the effective temperature is uncontrolled. Verify that `prototypes` are also L2-normalised (currently only `H_q_norm` and `proto_norm` are normalised before the dot product — this appears correct from the code, so H3 is lower priority).

### H4 — `intra_edge_ratio` Metric Bug Masking Real Signal

The metric is only computed for the last GNN layer (`if l == self.n_gnn_layers - 1`) using `support_mask`. In 1-shot setting (5 support nodes, 15 query nodes), there are only $5 \times 4 = 20$ possible support-support edge pairs out of $20 \times 19 = 380$ total edges. The `ss_mask` may be selecting too few edges for the ratio to register as non-zero in float32. Log the raw `same.sum()` and `ss_mask.sum()` counts explicitly.

---

## 6. What Would Make These Results Publishable

For the results to support a credible article, all of the following must be achieved:

### 6.1 Non-Negotiable Fixes

1. **Diagnose and fix the edge correction loss** (H1 above). The `intra_edge_ratio` must reach visibly positive values (≥0.3 within 50 epochs) before any other improvement matters.
2. **Raise FSAKE reproduced baseline to ~67–70%** on CIFAR-FS. This likely requires fixing the backbone training regime (H2). The reproduced baseline must be within 1–2% of the published FSAKE number when transferred to CIFAR-FS.
3. **DEKAE full must outperform FSAKE reproduced** by a statistically significant margin. Target: ≥2 pp improvement with the current 600-episode CI of ±1.05%.
4. **Topology recovery F1 of DEKAE must exceed kNN** at σ = 0.6–0.8. If it cannot beat 5-NN on the synthetic benchmark the structural contribution cannot be claimed.

### 6.2 Strong-to-Have for Acceptance

5. DEKAE must outperform A3 (edge projection alone), confirming the edge loss adds value.
6. Density curve must show stabilisation (not oscillation) over training epochs.
7. The variable-degree claim must be empirically demonstrated via the degree vs. difficulty plot (§7 of plan.md — currently not yet run).
8. Ablations need to run to the same epoch count as the main model (200 epochs, not 100).

### 6.3 Estimated Gap to Publishable State

| Metric | Current | Target |
|---|---|---|
| FSAKE reproduced accuracy | 43.4% | ~68–70% |
| DEKAE vs. FSAKE delta | −0.5 pp | ≥+2.0 pp |
| intra_edge_ratio at convergence | 0.000 | ≥0.30 |
| DEKAE topology recovery F1 vs kNN | −85% below kNN | ≥+10% above kNN |
| Density trajectory | Oscillating | Monotonically stabilising |

---

## 7. Positive Observations Worth Preserving

Despite the critical failures above, several engineering decisions are working correctly and should be retained:

- **CosineAnnealingLR**: the loss curves are smooth with no instability (gradient clipping + warm-up are effective — Risk 7 mitigated as designed).
- **Graph density ~0.08–0.09**: within the plan's "healthy" range; no topology collapse observed. The L1 sparsity constraint is functioning.
- **Top-k masking (E3) achieves the highest absolute accuracy** among sparsity ablations (43.55%); its stable density (0.063) demonstrates the collapse-prevention role described in Risk 1.
- **Computational cost**: ~22s/epoch at 300 episodes on T4 confirms the $O(N^2)$ complexity claim is valid in practice.
- **Checkpoint and history persistence**: the training infrastructure is production-ready.
- **A3 (edge-to-node projection) slightly beats FSAKE**: the +0.73 pp signal (44.14% vs 43.41%) is a positive structural finding, albeit marginal and covering only half the proposed novelty.

---

## 8. Recommended Next Steps

1. **Immediate (debugging, ~2 hours)**: Add diagnostic prints to verify `edge_correction_loss` produces non-zero values and that support-support pairs are being found. Fix any identified code path issue.
2. **Short-term (1–2 retraining runs, ~3–4 hours GPU)**: With edge loss confirmed working, retrain with all other settings unchanged and re-check `intra_edge_ratio` trajectory. If it rises, rerun the full 200-epoch training.
3. **Medium-term (full re-run, ~6 hours GPU)**: Once baseline FSAKE reproduction reaches ~68%, rerun both the main DEKAE model and all ablation groups to full 200 epochs.
4. **Validation (1 hour)**: Rerun the synthetic recovery experiment with the fixed trained model. If DEKAE now beats kNN F1, Contribution 3 becomes credible.
5. **Only then**: Proceed to Sections 6.3 (paired t-test), 6.5 (visualisations), and Group G (sensitivity analysis) as described in plan.md.

---

*All raw numbers sourced directly from notebook cell outputs. Benchmark comparisons derived from plan.md §4.2 and standard CIFAR-FS literature.*
