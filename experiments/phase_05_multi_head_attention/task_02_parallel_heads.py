from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


from experiments.phase_05_multi_head_attention.task_01_split_heads import q_reshaped, k_reshaped, v_reshaped

def manual_softmax(x):
    """I solved it on Tensortonic, so I use my own function😎"""
    max_val = torch.max(x, dim=-1, keepdim=True)[0]
    shifted = x - max_val
    exp_val = torch.exp(shifted)
    denominator = torch.sum(exp_val, dim=-1, keepdim=True)
    output = exp_val / denominator
    return output

#   q_reshaped @ k_reshaped.transpose(-2, -1)   # [2, 6, 4] x [2, 4, 6] =  [2, 6, 6]

head_dim = q_reshaped.shape[-1]     # 4

attention_output = manual_softmax((q_reshaped @ k_reshaped.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim))) @ v_reshaped

print("attention_output shape",attention_output.shape)