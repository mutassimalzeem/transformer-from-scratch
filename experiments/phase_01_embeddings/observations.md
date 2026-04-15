# Phase 1: Tokenization + Embeddings — Observations

## Task 1: Token to ID

- **Input:** `"the cat sat on the mat"` (string)
- **Output:** `[0, 1, 2, 3, 0, 4]` (list of int) — shape: `(seq_len,)`
- **What changed:** Raw text → dictionary lookup → integer IDs
- **What confused me:** Duplicate words in vocab created duplicate IDs. Also `id_to_word` was mapping `id → id` instead of `id → word`.
- **One thing I understood:** One word maps to one ID. The vocabulary is the bridge between text and numbers.

## Task 2: Embedding Lookup

- **Input:** `[0, 1, 2, 3, 0, 4]` (tensor of shape `(seq_len,)`)
- **Output:** tensor of shape `(seq_len, d_model)` → `(6, 8)`
- **With batch:** shape becomes `(1, 6, 8)` → `(batch_size, seq_len, d_model)`
- **What changed:** Sparse integer IDs → dense float vectors via `nn.Embedding` lookup
- **What confused me:** Why embeddings are initialized randomly and learned during training (not human-readable).
- **One thing I understood:** Each ID is an index into a lookup table. The table rows are the actual vectors — so `embedding(id)` just returns row `id` from the weight matrix.
