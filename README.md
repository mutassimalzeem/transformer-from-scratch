# Transformer From Scratch [![Phase 5 ✅](https://img.shields.io/badge/Progress-Phase%205%20Complete-brightgreen)](https://github.com/)

Learning Transformer architecture from scratch in small, beginner-friendly steps.

**Goal:** Deep understanding through shape-first implementations.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
cd experiments/phase_05_multi_head_attention
python task_01_split_heads.py   # [6,8] → [2,6,4]
python task_02_parallel_heads.py # Parallel attention
python task_03_concat_heads.py   # → [6,8] complete!
```

**Test all Phase 5:** `for f in task_*.py; do python $f; done`

## 📊 Progress: 5/8 Phases Complete

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| 1️⃣ Embeddings | ✅ | Token → embedding lookup |
| 2️⃣ Positional | ✅ | Position encoding added |
| 3️⃣ Single-head | ✅ | Basic self-attention |
| 4️⃣ QKV | ✅ | Scaled dot-product |
| **5️⃣ Multi-head** | **✅ NEW** | **Split → Parallel → Concat** |
| 6️⃣ Residual/LN/FFN | 🔄 50% | Tasks ready |
| 7️⃣ Encoder Block | ⏳ | - |
| 8️⃣ Decoder | ⏳ | - |

**Full checklist:** [roadmap.txt](roadmap.txt)

## 📁 Structure

```
experiments/
├── phase_01_embeddings/          # Vocab → embeddings
├── phase_02_positional_encoding/ # + position vectors  
├── phase_03_single_head_attention/
├── phase_04_qkv_attention/       # Q,K,V projections
├── phase_05_multi_head_attention/ # 👈 LATEST: [6,8] preserved!
├── phase_06_layernorm_residual_ffn/
└── ... decoder phases
```

**Docs per phase:** `observations.md` = shapes + insights

## 🎓 Learning Method

**For every task ask:**
1. **Input shape?**
2. **What operation?** 
3. **Output shape?**
4. **Why needed?**

## 📈 Shape Evolution (Phase 5)
```
[6,8] Q/K/V 
  ↓ split_heads
[2,6,4] heads  
  ↓ parallel_attn  
[2,6,4] outputs
  ↓ concat+project
[6,8] multi-head ✓
```

## 🔮 Next: Phase 6
- Residual connections
- Layer normalization  
- Feed-forward network

## 📚 Resources
- [utils/shapes.md](utils/shapes.md)
- [logs/progress.md](logs/progress.md) 
- [notes/transformer_notes.pdf](notes/transformer_notes.pdf)

## Run All Tests
```bash
# Phase 5 demo
cd experiments/phase_05_multi_head_attention && python task_03_concat_heads.py

# All phases (manual for now)
for p in phase_0{1..5}; do echo \"=== $p ===\"; cd experiments/$p && ls task_*.py; cd ../..; done
```

---
**Made with ❤️ for shape debugging and PyTorch experimentation**

