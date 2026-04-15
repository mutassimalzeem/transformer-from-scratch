import torch
import torch.nn as nn
from task_01_token_to_id import vocab, tokens

vocab_size = len(vocab)
d_model = 8             #   Embedding Vector size

embedding_layer = nn.Embedding(num_embeddings= vocab_size, embedding_dim= d_model)

token_tensor = torch.tensor(tokens)

embeddings = embedding_layer(token_tensor)

print("\nFirst token's vector representation:")
print(embeddings[0])

print("Single sentence shape:", embeddings.shape)   #   (sequence_length, d_model)

batch_embeddings = embeddings.unsqueeze(0)

print("Batch shape:", batch_embeddings.shape)   #   -> (batch_size, sequence_length, d_model)