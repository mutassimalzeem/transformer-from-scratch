from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos

d_model = 8

Wq = torch.nn.Linear(d_model, d_model)
Wk = torch.nn.Linear(d_model, d_model)
Wv = torch.nn.Linear(d_model, d_model)

q = Wq(embeddings_with_pos)
k = Wk(embeddings_with_pos)
v = Wv(embeddings_with_pos)


if __name__ == "__main__":
    print("Q shape:", q.shape)
    print("K shape:", k.shape)
    print("V shape:", v.shape)
    print("d_model:", d_model)
