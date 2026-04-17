from pathlib import Path
import sys
import torch.nn as nn

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

"""The standard Transformer expands the embedding dimension to a much larger space to learn complex features,
applies a non-linear activation curve, and then shrinks it back down to d_model so it can be passed to the next block."""

"""The Architecture
Linear Layer 1: Expands d_model to a wider dimension (the paper uses d_model*4).
Activation Function: Usually ReLU or GELU to introduce non-linearity (so the network can learn curves, not just straight lines).
Linear Layer 2: Shrinks the wider dimension back down to d_model."""

from experiments.phase_06_layernorm_residual_ffn.task_02_layer_norm import normed_output1
from experiments.phase_04_qkv_attention.task_01_make_qkv import d_model


# 1. Build the bundled FFN machinery
ffn_block = nn.Sequential(
    nn.Linear(d_model, d_model * 4),  # Expand: 8 -> 32
    nn.ReLU(),                        # Activate
    nn.Linear(d_model * 4, d_model)   # Shrink: 32 -> 8
)

# 2. Pass the normalized data through the FFN
ff_output_data = ffn_block(normed_output1)

print("FFN Output Shape:", ff_output_data.shape) # Should be [6, 8]