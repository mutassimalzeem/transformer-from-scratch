from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_05_multi_head_attention.task_03_concat_heads import final_output
from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos

#   Add the First Residual Connection --> Output = x + Attention(x)
output1 = embeddings_with_pos + final_output

print(output1.shape)