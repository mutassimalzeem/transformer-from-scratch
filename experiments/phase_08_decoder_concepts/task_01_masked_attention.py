from pathlib import Path
import sys
import torch
import torch.nn as nn

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Create causal mask so each position can only attend
        # to itself and previous tokens, not future tokens
        # Shape: [T, T]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        attn_output, attn_weights = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask
        )

        return attn_output, attn_weights


class DecoderMaskedAttentionDemo(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.masked_attention = MaskedSelfAttention(d_model, num_heads)

    def forward(self, token_ids):
        """
        token_ids: [B, T]
        """
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(0, seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Step 1: token embedding + positional embedding
        x = self.word_embedding(token_ids) + self.pos_embedding(positions)   # [B, T, d_model]

        # Step 2: masked self-attention
        output, attn_weights = self.masked_attention(x)

        return output, attn_weights


# Test
if __name__ == "__main__":
    vocab_size = 1000
    d_model = 8
    num_heads = 2
    max_seq_len = 20

    model = DecoderMaskedAttentionDemo(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    batch = torch.randint(0, vocab_size, (2, 6))   # [B=2, T=6]

    print("Input token ids shape:", batch.shape)

    output, attn_weights = model(batch)

    print("Masked attention output shape:", output.shape)      # [2, 6, 8]
    print("Attention weights shape:", attn_weights.shape)      # [2, 6, 6]
    print("Masked self-attention working!")