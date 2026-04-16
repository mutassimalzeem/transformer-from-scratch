import torch
if __name__ == "__main__":
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    from experiments.phase_03_single_head_attention.task_02_softmax_weights import attention_weights
    from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos, seq_len

    #   0.5 * money_vector + 0.3 * bank_vector + 0.2 * grows_vector
    
    contextual_embedding = []
    for i in range(seq_len):
        contextual_vector = torch.sum(attention_weights[i].unsqueeze(1) * embeddings_with_pos, dim=0)   #   unsqueeze(1) makes a [6, 1] column of weights
        contextual_embedding.append(contextual_vector)

    # Turn the list of 6 vectors back into a [6, 8] matrix
    final_context_tensor = torch.stack(contextual_embedding)

    print("Final shape:", final_context_tensor.shape) # Should be [6, 8]!