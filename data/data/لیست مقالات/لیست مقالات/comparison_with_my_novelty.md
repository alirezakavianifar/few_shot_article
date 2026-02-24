Based on your **two novelties** (from `merged_novelties.md`):

1. **Shallow–Deep Similarity + Beyond-Episode Memory (SDSimPoint-style idea)**
2. **Dynamic Edge-Driven Graph Connectivity (edge→node adaptive topology learning)**

and after carefully examining the four candidate articles:

* **SGE**
* **FSAKE**
* **MKGPL**
* **MPCF**

I will now give you a **clear, research-level recommendation** about which one is the strongest baseline for your novelties.

---

# 🔎 Step 1 — Match Each Paper Against Your Two Novelties

We evaluate structural alignment, not just topic similarity.

---

# 1️⃣ SGE (Semantic-Guided Generalization Enhancement)

### What it does:

* Uses CLIP + Stable Diffusion
* Cross-domain alternating training
* Semantic-guided augmentation
* Feature fusion at test time

### Alignment with Your Novelties:

| Your Novelty                           | Alignment with SGE         |
| -------------------------------------- | -------------------------- |
| Shallow–Deep similarity separation    | ❌ Not present             |
| Deep semantic modeling via interaction | ⚠️ Only via text prompts |
| Memory beyond episode                  | ❌ No                      |
| Dynamic graph connectivity             | ❌ No                      |
| Edge-driven topology                   | ❌ No                      |

### Verdict:

SGE is **semantic augmentation based**, not structural similarity modeling.
It is **conceptually different** from your work.

👉 **Not a good structural baseline.**

---

# 2️⃣ FSAKE (Few-shot Adaptive Neighbor Knowledge Embedding)

### What it does:

* Graph-level few-shot learning
* Neighbor-aware knowledge filtering
* Adaptive pooling
* Knowledge correction loss
* Uses Graph U-Net

### Alignment with Your Novelties:

| Your Novelty                       | Alignment with FSAKE              |
| ---------------------------------- | --------------------------------- |
| Graph structure modeling           | ✅ Yes                            |
| Neighbor-aware reasoning           | ✅ Yes                            |
| Structural adaptation              | ⚠️ Semi-dynamic (pooling-based) |
| Edge feature modeling              | ❌ No                             |
| Dynamic adjacency update per layer | ❌ No                             |
| Edge→node message passing         | ❌ No                             |
| Similarity taxonomy (shallow/deep) | ❌ No                             |

### Key Insight:

FSAKE:

* Uses **graph pooling**
* Still relies on node-based message passing
* Graph structure is not rebuilt dynamically via learned edge embeddings

Your novelty:

* Reconstructs adjacency from learned edge features
* Edge-driven topology learning
* Dynamic connectivity
* Potential variable degree graph

This is **strictly more advanced than FSAKE structurally**.

### Verdict:

👉 **Best baseline for your Dynamic Graph Connectivity novelty.**

It is the closest structural neighbor in the literature.

---

# 3️⃣ MKGPL (Graph Prompt Learning with Multi-View Knowledge)

### What it does:

* Positive text graph
* Negative text graph
* Image graph
* Graph prompts
* Multi-view fusion

### Alignment with Your Novelties:

| Your Novelty                       | Alignment             |
| ---------------------------------- | --------------------- |
| Graph modeling                     | ✅ Yes                |
| Multi-view semantic similarity     | ⚠️ Yes (text-based) |
| Dynamic topology                   | ❌ No                 |
| Edge-driven updates                | ❌ No                 |
| Similarity taxonomy (shallow/deep) | ❌ Not explicit       |
| Cross-episode memory               | ❌ No                 |

MKGPL is:

* Graph-structured prompt learning
* Focused on positive/negative semantics
* Based on CLIP adaptation

Your method is:

* Structural graph learning
* Edge-feature–driven topology evolution
* Dynamic rewiring

Conceptual gap is large.

👉 Not a suitable baseline for structural novelty.

---

# 4️⃣ MPCF (Multimodal Prototypical Networks with Co-Metric Fusion)

### What it does:

* Image + text prototypes
* Contrastive alignment
* Weighted metric fusion
* Cross-domain few-shot HSI

### Alignment:

| Your Novelty             | Alignment           |
| ------------------------ | ------------------- |
| Similarity modeling      | ✅ Yes              |
| Deep semantic fusion     | ⚠️ Yes (via text) |
| Shallow–deep separation | ❌ No               |
| Graph modeling           | ❌ No               |
| Dynamic connectivity     | ❌ No               |
| Edge modeling            | ❌ No               |

This is metric fusion, not structural innovation.

👉 Not a structural baseline.

---

# 🏆 Final Ranking (Best Baseline for Your Work)

| Rank | Paper           | Suitability |
| ---- | --------------- | ----------- |
| 🥇 1 | **FSAKE** | ⭐⭐⭐⭐⭐  |
| 🥈 2 | MKGPL           | ⭐⭐        |
| 🥉 3 | MPCF            | ⭐          |
| ❌ 4 | SGE             | ❌          |

---

# 🎯 Final Answer

## ✅ **FSAKE should be used as the baseline for your novelties.**

Here is why:

### 1️⃣ It is graph-based few-shot learning.

Your second novelty is fundamentally a graph innovation.

### 2️⃣ It performs structural reasoning.

But:

* It uses static adjacency.
* It does not learn edge features explicitly.
* It does not reconstruct topology dynamically.

Your method **strictly generalizes and improves over FSAKE**.

---

# 🔬 How You Should Position Your Novelty Against FSAKE

In a paper, you can claim:

> While FSAKE improves node importance selection using neighbor-aware filtering and correction supervision, it relies on fixed graph connectivity and node-centric message passing. In contrast, our method introduces an edge-feature–driven dynamic topology learning mechanism that reconstructs graph structure adaptively at each layer, enabling edge-centric relational modeling and variable-degree connectivity.

That is a strong novelty positioning.

---

# 📌 What About Your Shallow–Deep Similarity Novelty?

If you are positioning that one:

None of the four papers explicitly model similarity taxonomy.

So:

* You could compare against FSAKE structurally.
* But your shallow–deep similarity novelty is orthogonal to all four.

---

# 🧠 Strategic Advice

If your goal is:

### 🟢 Publishing in a graph-oriented journal:

Use **FSAKE** as baseline.

### 🟢 Publishing in a multimodal/semantic journal:

Then MKGPL or MPCF might be comparison references, but not baselines.

---

# 🏁 Final Conclusion

✔ For your **Dynamic Graph Connectivity novelty → FSAKE is the correct baseline.**

✔ For your **Shallow–Deep Similarity novelty → none of the four is a direct baseline**, but FSAKE remains the closest structural comparison.

---

If you want, I can next:

* Help you write the exact “Baseline Justification” paragraph for your paper.
* Or analyze whether combining both of your novelties makes the contribution stronger than FSAKE-level novelty.
