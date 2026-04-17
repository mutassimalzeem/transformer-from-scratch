# Transformer From Scratch [![Phase 7 ✅](https://img.shields.io/badge/Progress-Phase%207%20Complete-brightgreen)](https://github.com/)

Learning Transformer architecture from scratch in small, beginner-friendly steps.

**Goal:** Deep understanding through shape-first implementations.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
cd experiments/phase_07_encoder_block
python task_01_build_encoder_block.py  # token_ids [2,6] → [2,6,8] FULL ENCODER!
```

**Test Phase 7:** `python task_01_build_encoder_block.py`

## 📊 Progress: 7/8 Phases Complete

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| 1️⃣ Embeddings | ✅ | Token → embedding lookup |
| 2️⃣ Positional | ✅ | Position encoding added |
| 3️⃣ Single-head | ✅ | Basic self-attention |
| 4️⃣ QKV | ✅ | Scaled dot-product |
| 5️⃣ Multi-head | ✅ | Split → Parallel → Concat |
| 6️⃣ Residual/LN/FFN | ✅ | Residual → Norm → FFN → Norm |
| **7️⃣ Encoder Block** | **✅ NEW** | **Modular nn.Module encoder!** |
| 8️⃣ Decoder | ⏳ | - |

**Full checklist:** [roadmap.txt](roadmap.txt)

## 📁 Structure

```
experiments/
├── phase_01-06/                 # Building blocks
├── phase_07_encoder_block/      # 👈 LATEST: FULL ENCODER!
└── phase_08_decoder_concepts/   # Final frontier
```

## 📈 Shape Evolution (Phase 7)
```
token_ids [B,T] e.g. [2,6]
  ↓ embed + pos embed
[B,T,d_model] [2,6,8]
  ↓ EncoderBlock (all phases 3-6)
[B,T,d_model] [2,6,8] ✓
```

## 🎓 Learning Method
**4 Questions Per Task:**
1. Input shape?
2. Operation? 
3. Output shape?
4. Why needed?

## 🔮 Next: Phase 8 Decoder
- Masked attention
- Cross-attention

## 📚 Resources
- [utils/shapes.md](utils/shapes.md)
- [logs/progress.md](logs/progress.md)
- All `observations.md` files

## Run All Tests
```bash
# Phase 7 full encoder
cd experiments/phase_07_encoder_block && python task_01_build_encoder_block.py

# Summary
for p in phase_0{1..7}; do echo \"=== $p ===\" && ls experiments/$p/task_*.py; done
```

---
**7/8 phases: Encoder complete! Ready for decoder.** ❤️

