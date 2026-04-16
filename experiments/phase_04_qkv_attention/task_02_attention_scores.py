from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_04_qkv_attention.task_01_make_qkv import q, k

attention_score = q @ k.T

if __name__ == "__main__":
    print("Attention Scores Shape:", attention_score.shape)
