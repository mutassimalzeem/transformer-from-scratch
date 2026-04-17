from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_06_layernorm_residual_ffn.task_01_residual_add import output1
from experiments.phase_04_qkv_attention.task_01_make_qkv import d_model


# 1. Build the layer
norm1_layer = torch.nn.LayerNorm(d_model)

# 2. Pass the data through the layer
normed_output1 = norm1_layer(output1)

print("Norm 1 Shape:", normed_output1.shape) # Should be [6, 8]