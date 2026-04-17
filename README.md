# Transformer From Scratch [![Phase 6 ✅](https://img.shields.io/badge/Progress-Phase%206%20Complete-brightgreen)](https://github.com/)

Learning Transformer architecture from scratch in small, beginner-friendly steps.

**Goal:** Deep understanding through shape-first implementations.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
cd experiments/phase_06_layernorm_residual_ffn
python task_01_residual_add.py     # x + attn(x)
python task_02_layer_norm.py       # Stabilize [6,8]  
python task_03_feed_forward.py     # Expand 8→32→8 ReLU
python task_04_Add_and_Norm_2.py   # Full encoder layer!
```

**Test Phase 6:** `for f in task_*.py; do python $f; done`

## 📊 Progress: 6/8 Phases Complete

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| 1️⃣ Embeddings | ✅ | Token → embedding lookup |
| 2️⃣ Positional | ✅ | Position encoding added |
| 3️⃣ Single-head | ✅ | Basic self-attention |
| 4️⃣ QKV | ✅ | Scaled dot-product |
| 5️⃣ Multi-head | ✅ | Split → Parallel → Concat |
| **6️⃣ Residual/LN/FFN** | **✅ NEW** | **Residual → Norm → FFN → Norm** |
| 7️⃣ Encoder Block | 🔄 | - |
| 8️⃣ Decoder | ⏳ | - |

**Full checklist:** [roadmap.txt](roadmap.txt)

## 📁 Structure

```
experiments/
├── phase_01_embeddings/          
├── phase_02_positional_encoding/ 
├── phase_03_single_head_attention/
├── phase_04_qkv_attention/       
├── phase_05_multi_head_attention/
├── phase_06_layernorm_residual_ffn/ # 👈 LATEST: Full encoder internals!
└── ... decoder phases
```

**Docs per phase:** `observations.md` = shapes + insights

## 🎓 Learning Method

**For every task ask:**
1. **Input shape?**
2. **What operation?** 
3. **Output shape?**
4. **Why needed?**

## 📈 Shape Evolution (Phase 6)
```
Multihead [6,8]
  ↓ residual+add
[6,8] + attn    
  ↓ LayerNorm1
Normed [6,8]    
  ↓ FFN (8→32→8)
FFN [6,8]       
  ↓ residual+add+Norm2
Encoder layer [6,8] ✓
```

## 🔮 Next: Phase 7
- Full encoder block assembly

## 📚 Resources
- [utils/shapes.md](utils/shapes.md)
- [logs/progress.md](logs/progress.md) 
- [notes/transformer_notes.pdf](notes/transformer_notes.pdf)

## Run All Tests
```bash
# Phase 6 demo (full layer internals)
cd experiments/phase_06_layernorm_residual_ffn && python task_04_Add_and_Norm_2.py

# Phase summary
for p in phase_0{1..6}; do echo \"=== $p ===\" && cd experiments/$p && ls task_*.py && cd ../..; done
```

---
**Made with ❤️ for shape debugging and PyTorch experimentation**

