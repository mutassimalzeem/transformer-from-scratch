# Phase 6: Residuals + LayerNorm + FFN — Observations

## Task 1: Residual Add (Attention)

- **Input:** `embeddings_with_pos` + `final_output` (multi-head) both `(6, 8)`
- **Output:** `output1` `(6, 8)`
- **What changed:** `x + Attention(x)` — **first residual connection** adds attention output back to input
- **What confused me:** Why add before norm? (Pre-norm vs post-norm variants exist)
- **One thing I understood:** Residuals create **identity shortcut** → gradients flow directly, prevents vanishing gradients in deep stacks.

## Task 2: Layer Normalization 1

- **Input:** `output1` `(6, 8)`
- **Output:** `normed_output1` `(6, 8)`
- **What changed:** `LayerNorm(d_model)` normalizes across features **per token** (mean=0, std=1)
- **What confused me:** Normalizes last dim (features), not sequence — each position independent.
- **One thing I understood:** **Stabilizes training** by reducing internal covariate shift. Input to FFN now **normalized**.

## Task 3: Feed Forward Network

- **Input:** `normed_output1` `(6, 8)`
- **Output:** `ff_output_data` `(6, 8)`
- **What changed:** `Linear(8→32) → ReLU → Linear(32→8)` — **position-wise MLP**
- **What confused me:** Why expand 4x then shrink? Creates high-dim space for complex feature mixing.
- **One thing I understood:** **Identical FFN per position** (unlike attention). Non-linearity (ReLU) enables learning curves.

## Task 4: Add & Norm 2 (Final)

- **Input:** `ff_output_data` + `normed_output1` `(6, 8)`
- **Output:** `final_encoder_output` `(6, 8)` — **complete encoder layer!**
- **What changed:** Second residual `normed_output1 + FFN()` → final `LayerNorm`
- **What confused me:** Two norms per layer — one before attention (not shown), one before/after FFN.
- **One thing I understood:** **Full block**: `Norm1 → Attn → Add → Norm2 → FFN → Add`. Shape-stable [6,8] → richer representations!
