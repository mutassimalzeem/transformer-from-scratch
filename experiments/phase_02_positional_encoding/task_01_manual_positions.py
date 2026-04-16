import numpy as np

def get_positional_encoding_fast(max_seq_len, d_model):
    """Vectorized positional encoding — same result, way faster."""

    # Position indices: [0, 1, 2, ..., max_seq_len-1] → shape (max_seq_len, 1)
    positions = np.arange(max_seq_len)[:, np.newaxis]

    # Dimension indices: [0, 2, 4, ..., d_model-2] → shape (1, d_model//2)
    dims = np.arange(0, d_model, 2)[np.newaxis, :]

    # Compute all angles at once
    angles = positions / (10000 ** (dims / d_model))

    # Allocate output
    PE = np.zeros((max_seq_len, d_model))

    # Fill even columns with sin, odd columns with cos
    PE[:, 0::2] = np.sin(angles)  # 0, 2, 4, ...
    PE[:, 1::2] = np.cos(angles)  # 1, 3, 5, ...

    return PE

print(get_positional_encoding_fast(50, 64).shape)