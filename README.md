# Transformer From Scratch

Learning the Transformer architecture from scratch in small, beginner-friendly steps.

This repository is not about copying a full implementation directly.  
The goal is to understand each building block deeply by implementing it phase by phase.

## Learning Goal

Build intuition first, then code.

Topics covered in this journey:

- Tokenization and vocabulary
- Embedding lookup
- Positional encoding
- Basic self-attention intuition
- Query, Key, Value (QKV)
- Scaled dot-product attention
- Multi-head attention
- Residual connections
- Layer normalization
- Feed-forward network
- Encoder block
- Masked self-attention
- Cross-attention
- Decoder concepts

---

## Project Structure

```text
transformer-from-scratch/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notes/
│   └── transformer_notes.pdf
│
├── experiments/
│   ├── phase_01_embeddings/
│   │   ├── task_01_token_to_id.py
│   │   ├── task_02_embedding_lookup.py
│   │   └── observations.md
│   │
│   ├── phase_02_positional_encoding/
│   │   ├── task_01_manual_positions.py
│   │   ├── task_02_add_position_to_embedding.py
│   │   └── observations.md
│   │
│   ├── phase_03_single_head_attention/
│   │   ├── task_01_similarity_scores.py
│   │   ├── task_02_softmax_weights.py
│   │   ├── task_03_weighted_sum.py
│   │   └── observations.md
│   │
│   ├── phase_04_qkv_attention/
│   │   ├── task_01_make_qkv.py
│   │   ├── task_02_attention_scores.py
│   │   ├── task_03_scaled_attention.py
│   │   └── observations.md
│   │
│   ├── phase_05_multi_head_attention/
│   │   ├── task_01_split_heads.py
│   │   ├── task_02_parallel_heads.py
│   │   ├── task_03_concat_heads.py
│   │   └── observations.md
│   │
│   ├── phase_06_layernorm_residual_ffn/
│   │   ├── task_01_residual_add.py
│   │   ├── task_02_layer_norm.py
│   │   ├── task_03_feed_forward.py
│   │   └── observations.md
│   │
│   ├── phase_07_encoder_block/
│   │   ├── task_01_build_encoder_block.py
│   │   └── observations.md
│   │
│   └── phase_08_decoder_concepts/
│       ├── task_01_masked_attention.py
│       ├── task_02_cross_attention.py
│       └── observations.md
│
├── utils/
│   ├── shapes.md
│   └── helper_notes.md
│
└── logs/
    └── progress.md

```

##    How I am learning

For every task, I try to answer 4 things:

What is the input shape?
What operation is happening?
What is the output shape?
Why is this needed?

This keeps the focus on understanding instead of blindly coding.



##    Progress Roadmap
Phase 1 — Embeddings
 Create manual vocabulary
 Convert sentence to token ids
 Perform embedding lookup
 Inspect output shape


Phase 2 — Positional Encoding
 Create position indices
 Build positional vectors
 Add positional encoding to embeddings
 Compare before vs after


Phase 3 — Basic Self-Attention
 Compute similarity scores
 Convert scores to weights
 Compute weighted sum
 Interpret contextual embedding


Phase 4 — QKV Attention
 Create Query, Key, Value
 Compute attention score matrix
 Apply scaling
 Apply softmax
 Multiply with Value


Phase 5 — Multi-Head Attention
 Split into heads
 Run attention per head
 Concatenate heads
 Final projection


Phase 6 — Residual + LayerNorm + FFN
 Residual connection
 Layer normalization
 Feed-forward network
 Add + Norm again


Phase 7 — Encoder Block
 Combine all encoder components
 Test on toy input
 Verify shapes at every step


Phase 8 — Decoder Concepts
 Masked self-attention
 Cross-attention
 Decoder intuition
 Inference intuition


##    Notes Format
Each phase contains an observations.md file.
For every task, I write:

Input shape
Output shape
What changed
What confused me
What I understood


Run
Create environment and install dependencies:
```bash
pip install -r requirements.txt
```

Run a task file:
```bash
python experiments/phase_01_embeddings/task_01_token_to_id.py
```

##    Why this repo exists

The purpose of this repository is to turn theoretical Transformer knowledge into practical understanding through small implementations and shape-based reasoning.

##    Future Improvements
- Add PyTorch module versions after manual implementations
- Add visualization notebooks
- Add attention heatmaps
- Add mini encoder-decoder project
- Add toy next-token prediction example
