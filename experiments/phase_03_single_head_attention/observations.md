# Phase 3: Single-Head Attention Intuition — Observations

## Task 1: Similarity Scores

- **Input:** `embeddings_with_pos` shape `(seq_len, d_model)` → `(6, 8)`
- **Output:** `raw_score` shape `(6, 6)` — each word's similarity to all words
- **What changed:** Vector dot products (`@ .T`) create **attention matrix** where rows = queries, columns = keys
- **What confused me:** Why use full vocab embeddings (8) but similarity only among sequence positions (6)?
- **One thing I understood:** **Each word looks at EVERY word** via pairwise similarities. Higher score = more similar meaning/position/context.

## Task 2: Softmax Weights

- **Input:** `raw_score` `(6, 6)`
- **Output:** `attention_weights` `(6, 6)` — probabilities per row summing to 1.0
- **What changed:** Raw logits → normalized weights via **stable softmax** (max subtract prevents overflow)
- **What confused me:** Why `keepdim=True`? (To preserve shape for broadcasting/matrix ops)
- **One thing I understood:** **Weights sum to 1** across each row → creates probability distribution. Word 0's attention: ~how much it 'attends' to each position!

## Task 3: Weighted Sum

- **Input:** `attention_weights` `(6, 6)` + `embeddings_with_pos` `(6, 8)`
- **Output:** `contextual_embedding` list → intended tensor `(6, 8)` of context-aware vectors
- **What changed:** Element-wise multiply + sum → **each position's embedding becomes weighted average** of ALL embeddings
- **What confused me:** Broadcasting shapes `[6] * [6,8]` → need `unsqueeze(-1)` or matmul for `[6,1,8]`
- **One thing I understood:** **Output is context-aware!** New vector for position i mixes info from similar positions proportional to attention weights. Foundation of self-attention! ✓

