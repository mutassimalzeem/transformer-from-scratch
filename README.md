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