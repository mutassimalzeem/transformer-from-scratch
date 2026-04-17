from pathlib import Path
import sys
import torch.nn as nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.ReLU(), 
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Multi-head attention sublayer
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x)  # [B,T,d_model]
        # Residual + norm
        x_norm1 = self.norm1(x + attn_output)
        
        # FFN sublayer  
        ffn_output = self.ffn(x_norm1)  # [B,T,d_model]
        # Residual + norm
        output = self.norm2(x_norm1 + ffn_output)
        
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)
        self.encoder_block = EncoderBlock(d_model, num_heads)

    def forward(self, token_ids):
        # token_ids: [B, T]
        batch_size, seq_len = token_ids.size()
        positions = torch.arange(0, seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed + pos [B,T,d_model]
        x = self.word_embedding(token_ids) + self.pos_embedding(positions)
        
        # Encoder block
        final_context = self.encoder_block(x)
        
        return final_context

# Test
if __name__ == '__main__':
    model = TransformerEncoder(vocab_size=1000, d_model=8, num_heads=2, max_seq_len=100)
    batch = torch.randint(0, 1000, (2, 6))  # [B=2, T=6]
    print('Input shape:', batch.shape)
    output = model(batch)
    print('Encoder output shape:', output.shape)  # [2,6,8]
    print('Encoder block working!')

