# Phase 5: Multi-Head Attention — Observations

## Task 1: Split Heads

- **Input:** `q, k, v` `(6, 8)` from Phase 4 QKV
- **Output:** `q_reshaped` `(2, 6, 4)` (num_heads, seq_len, head_dim)
- **What changed:** `reshape(6,2,4).transpose(0,1)` splits embedding dim into parallel heads
- **What confused me:** Transpose after reshape — why [seq_len,num_heads,head_dim] → [num_heads,seq_len,head_dim]?
- **One thing I understood:** Each head gets **narrower** view (head_dim=d_model/num_heads). Allows attending to different subspaces in parallel.

## Task 2: Parallel Heads

- **Input:** Head-split Q,K,V `(2,6,4)`
- **Output:** `attention_output` `(2,6,4)` 
- **What changed:** **Simultaneous attention** per head: q@k.T/√head_dim → softmax → @v (all heads independent)
- **What confused me:** Custom softmax implementation (but stable w/ max subtraction).
- **One thing I understood:** Heads compute **in parallel** → same computation but different subspaces = diverse attention patterns combined later.

## Task 3: Concat Heads

- **Input:** Head outputs `(2,6,4)`
- **Output:** `final_output` `(6, 8)` — back to d_model!
- **What changed:** `transpose(0,1)` → `contiguous().reshape(6,8)` → `nn.Linear(8,8)` mixes heads
- **What confused me:** `contiguous()` requirement after transpose for efficient reshape.
- **One thing I understood:** **Final projection** crucial — concatenating raw heads alone loses inter-head mixing. Linear learns optimal combination!
