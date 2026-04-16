# Phase 4: QKV Attention — Observations

## Task 1: Make QKV

- **Input:** `embeddings_with_pos` `(6, 8)`
- **Output:** `q, k, v` each shape `(6, 8)`
- **What changed:** Three identical `nn.Linear(8,8)` projections create **learned** query/key/value representations from positional embeddings
- **What confused me:** Linear layers have random weights each run → q/k/v change, but shapes stable. (Training learns optimal projections!)
- **One thing I understood:** **Same input → three different views** (what I seek=q, what I have for matching=k, what to retrieve=v). Dot-product now q@k instead of embedding@embedding.

## Task 2: Attention Scores

- **Input:** `q, k` `(6,8)`
- **Output:** `attention_score` `(6,6)`
- **What changed:** `q @ k.T` → similarity **between projected representations**
- **What confused me:** Why not transpose k dims? Shape math: q[6,8] @ k.T[8,6] → [6,6]
- **One thing I understood:** **Separation of concerns**: Raw similarities (Phase 3) → **learnable** via Wq/Wk. More flexible!

## Task 3: Scaled Attention

- **Input:** `attention_score` `(6,6)`, `v` `(6,8)`
- **Output:** `output` `(6,8)` — full attention mechanism!
- **What changed:** `softmax(scores / sqrt(d_model)) @ v` → stable + value-weighted context
- **What confused me:** Import chains (task_03 needs task_02/task_01/manual_softmax fixed).
- **One thing I understood:** **Scaling prevents softmax saturation** (dot-products grow with dim). Complete self-attn: input [6,8] → output [6,8] context-aware! Ready for multi-head.

