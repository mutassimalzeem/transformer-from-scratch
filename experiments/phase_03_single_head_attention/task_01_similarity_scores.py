from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos
import numpy as np
import torch

#   target_vector = embeddings_with_pos[2]

raw_score = embeddings_with_pos @ embeddings_with_pos.T

# 2. Shape tracking (matches your learning framework)
print("Input embeddings shape:", embeddings_with_pos.shape)
print("Similarity matrix (raw_scores) shape:", raw_score.shape)

# 3. Inspect the actual scores
print("\nFirst row (similarity of word 0 with all words):")
print(raw_score[0])