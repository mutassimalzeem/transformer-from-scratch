from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_05_multi_head_attention.task_02_parallel_heads import attention_output
from experiments.phase_04_qkv_attention.task_01_make_qkv import d_model

#  [2, 6, 4] -> [6, 2, 4]
context_transposed = attention_output.transpose(0, 1)

# contiguous AFTER transposing, then reshape to [6, 8]
context_concatenated = context_transposed.contiguous().reshape(6, 8)
print("Concatenated Shape:", context_concatenated.shape) # [6, 8]

# Final Linear Projection
# This mixes the independent head insights together
w0 = torch.nn.Linear(d_model, d_model)
final_output = w0(context_concatenated)

print("Final Output Shape:", final_output.shape) # BOOM: [6, 8]