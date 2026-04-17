from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos
from experiments.phase_04_qkv_attention.task_01_make_qkv import d_model, q, k, v

#   q, k, v dimensions need to be [seq_len, num_heads, head_dim] [6, 8] --> [6, 2, 4] --> [2, 6, 4]

#   reshape q, k, v to change your [6, 8] tensor into [6, 2, 4] [seq_len, num_heads, head_dim] --> transpose(0, 1) --> [num_heads, seq_len, head_dim]

q_reshaped = torch.reshape(q, (6, 2, 4)).transpose(0,1)
k_reshaped = torch.reshape(k, (6, 2, 4)).transpose(0,1)
v_reshaped = torch.reshape(v, (6, 2, 4)).transpose(0,1)


print("Q SHape:" ,q.shape)
print("Q Re-Sape:" ,q_reshaped.shape)
