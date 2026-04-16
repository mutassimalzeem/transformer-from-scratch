# Phase 2: Positional Encoding — Observations

## Task 1: Manual Positions (Vectorized Positional Encoding)

- **Input:** `max_seq_len=50, d_model=64` (scalars)
- **Output:** `PE` array of shape `(max_seq_len, d_model)` → `(50, 64)`
- **What changed:** Scalar parameters → dense position vectors using sin/cos formula with geometric progression (`10000^(2i/d_model)`)
- **What confused me:** Initially thought positions needed manual loops, but vectorized `np.arange` + broadcasting computes **all** positions/dimensions in one go.
- **One thing I understood:** Each position gets a **unique** vector pattern across dimensions. Even/odd dims use sin/cos → creates smooth, periodic signals distinguishable by Transformers.

## Task 2: Add Position to Embedding

- **Input:** 
  - Token embeddings: `(seq_len, d_model)` → `(6, 8)`
  - Positional encodings: `(seq_len, d_model)` → `(6, 8)`
- **Output:** `embeddings + pe_tensor` → still `(6, 8)` (same shape!)
- **What changed:** **Pure element-wise addition** — each position's embedding gets its positional "offset" added directly
- **What confused me:** Why addition works? (Answer: Linear models can learn to separate word+position signals since they're orthogonal-ish)
- **One thing I understood:** **Transformers are POSITION-agnostic by design** (CNN/RNN have baked-in order). Positional encoding injects order via this simple addition trick → genius!
