from pathlib import Path
import sys
import math

if __name__ == "__main__" and __package__ is None:
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import math

import torch

def manual_softmax(x):
    """I solved it on Tensortonic, so I use my own function😎"""
    max_val = torch.max(x, dim=-1, keepdim=True)[0]
    shifted = x - max_val
    exp_val = torch.exp(shifted)
    denominator = torch.sum(exp_val, dim=-1, keepdim=True)
    output = exp_val / denominator
    return output

from experiments.phase_04_qkv_attention.task_01_make_qkv import v, d_model
from experiments.phase_04_qkv_attention.task_02_attention_scores import attention_score

scaled_score = manual_softmax(attention_score / math.sqrt(d_model))
output = scaled_score @ v

if __name__ == "__main__":
    print("Scaled attention output shape:", output.shape)

