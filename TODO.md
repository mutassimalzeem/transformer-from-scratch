# Phase 4 QKV Fixes - TODO

## Approved Plan Steps:
- [x] Step 1: Edit `experiments/phase_04_qkv_attention/task_01_make_qkv.py` - Move q, k, v, d_model computations to module top-level for imports
- [x] Step 2: Test run: `cd experiments/phase_04_qkv_attention && python task_01_make_qkv.py`
- [x] Step 3: Test `python task_02_attention_scores.py` - Verify attention_score prints
- [x] Step 4: Test `python task_03_scaled_attention.py` - Verify scaled attention completes ✓ Output [6,8]
- [x] Step 5: Fill observations.md (Phase 4)
- [x] Step 6: Mark complete, update progress logs if needed

**Phase 4 COMPLETE!** All tasks run successfully: QKV projections → scores → scaled dot-product attention.
Test anytime: `cd experiments/phase_04_qkv_attention && python task_0[1-3]_*.py`


