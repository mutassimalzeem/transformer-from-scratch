# Phase 7: Encoder Block — Observations

## Task 1: Build Encoder Block

- **Input:** `token_ids` `[B, T]` (e.g. [2, 6])
- **Output:** `final_context` `[B, T, d_model]` (e.g. [2, 6, 8])
- **What changed:** **Full stack** — Embed+Pos → EncoderBlock(Attn→add/norm→FFN→add/norm)
- **What confused me:** `nn.MultiheadAttention` returns tuple (output, weights); needs `[0]` or `_`. `batch_first=True` for [B,T,d].
- **One thing I understood:** **Modular nn.Module classes** assemble phases 1-6 into reusable encoder. Learned pos embeddings + full forward pass working!
