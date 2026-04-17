# Phase 8: Decoder Concepts — Observations\n\n## 

##  Task 1: Masked Self-Attention\n\n- 

-   **Input:** `token_ids` `(2, 6)` \n
-   **Output:** `output` `(2, 6, 8)`, `weights` `(2, 6, 6)`\n
-   **What changed:** `torch.triu` causal mask blocks future tokens via `attn_mask=mask`\n-
-   **What confused me:** Mask is `(T,T)` but applies across batch `(B, T, T)` heads\n -    **One thing I understood:** Autoregressive decoding — each position sees only past/future-masked self-attention\n\n```python\nmask = torch.triu(torch.ones(T, T), diagonal=1).bool()\n# attn_mask prevents position i seeing j > i\n```\n\n## Task 2: Cross-Attention\n\n- **Input:** encoder_tokens `(2, 7)`, decoder_tokens `(2, 5)` \n- **Output:** `output` `(2, 5, 8)`, `weights` `(2, 5, 7)`\n- **What changed:** `query=decoder_x`, `key/value=encoder_output` — decoder queries full encoder context\n- **What confused me:** Separate src/tgt embeddings (different vocabs multilingual)\n- **One thing I understood:** **Translation bridge** — decoder attends entire source while generating target sequentially\n\n```python\n# decoder_x @ encoder_output → decoder sees source context\nattn_output, weights = multihead_attn(query=decoder_x, key=encoder, value=encoder)\n```\n\n**Decoder intuition:** Masked self (past context) + cross (source context) → complete decoder layer!
