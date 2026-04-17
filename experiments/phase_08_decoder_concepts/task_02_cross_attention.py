from pathlib import Path
import sys
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, decoder_x, encoder_output):
        """
        decoder_x:      [B, T_dec, d_model]   -> Query
        encoder_output: [B, T_enc, d_model]   -> Key, Value

        returns:
        output: [B, T_dec, d_model]
        """
        attn_output, attn_weights = self.multihead_attn(
            query=decoder_x,
            key=encoder_output,
            value=encoder_output
        )

        return attn_output, attn_weights


class CrossAttentionDemo(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, max_seq_len):
        super().__init__()

        # Encoder side embeddings
        self.src_word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Decoder side embeddings
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.cross_attention = CrossAttention(d_model, num_heads)

    def forward(self, encoder_token_ids, decoder_token_ids):
        """
        encoder_token_ids: [B, T_enc]
        decoder_token_ids: [B, T_dec]
        """
        batch_size, src_len = encoder_token_ids.shape
        _, tgt_len = decoder_token_ids.shape

        # Source positions
        src_positions = torch.arange(0, src_len, device=encoder_token_ids.device)
        src_positions = src_positions.unsqueeze(0).expand(batch_size, src_len)

        # Target positions
        tgt_positions = torch.arange(0, tgt_len, device=decoder_token_ids.device)
        tgt_positions = tgt_positions.unsqueeze(0).expand(batch_size, tgt_len)

        # Encoder representations
        encoder_output = self.src_word_embedding(encoder_token_ids) + self.src_pos_embedding(src_positions)
        # [B, T_enc, d_model]

        # Decoder representations
        decoder_x = self.tgt_word_embedding(decoder_token_ids) + self.tgt_pos_embedding(tgt_positions)
        # [B, T_dec, d_model]

        # Cross-attention: decoder queries encoder output
        output, attn_weights = self.cross_attention(decoder_x, encoder_output)

        return output, attn_weights


# Test
if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    d_model = 8
    num_heads = 2
    max_seq_len = 20

    model = CrossAttentionDemo(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    encoder_batch = torch.randint(0, src_vocab_size, (2, 7))   # [B=2, T_enc=7]
    decoder_batch = torch.randint(0, tgt_vocab_size, (2, 5))   # [B=2, T_dec=5]

    print("Encoder token ids shape:", encoder_batch.shape)
    print("Decoder token ids shape:", decoder_batch.shape)

    output, attn_weights = model(encoder_batch, decoder_batch)

    print("Cross-attention output shape:", output.shape)        # [2, 5, 8]
    print("Attention weights shape:", attn_weights.shape)       # [2, 5, 7]
    print("Cross-attention working!")