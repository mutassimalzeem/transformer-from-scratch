import torch
from pathlib import Path
import sys

if __name__ == "__main__":
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    from experiments.phase_03_single_head_attention.task_01_similarity_scores import raw_score

    print("Raw scores shape:", raw_score.shape)  # Should be [6, 8]
    
    def manual_softmax(x):
        """
        Numerically stable softmax along the LAST dimension only
        """
        # Step 1: Subtract max for numerical stability (along last dim)
        max_val = torch.max(x, dim=-1, keepdim=True)[0]  # Keep dimension!
        print("Max shape:", max_val.shape)  # Should be [6, 1]
        
        shifted = x - max_val
        
        # Step 2: Apply exponential
        exp_val = torch.exp(shifted)
        
        # Step 3: Sum ALONG LAST DIMENSION ONLY (keepdim=True preserves shape)
        denominator = torch.sum(exp_val, dim=-1, keepdim=True)  # Shape: [6, 1]
        print("Denominator shape:", denominator.shape)
        
        # Step 4: Normalize
        output = exp_val / denominator
        
        return output

    attention_weights = manual_softmax(raw_score)
    
    print("\n✓ Attention Weights Shape:", attention_weights.shape)  # Should be [6, 8]
    print("\nFirst row (word 0's attention distribution):")
    print(attention_weights[0])
    print("\nSum of first row (should be 1.0):", attention_weights[0].sum().item())
    print("\nAll rows sum to 1.0?", torch.allclose(attention_weights.sum(dim=-1), torch.ones(6)))