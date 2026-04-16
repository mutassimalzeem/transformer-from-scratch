import numpy as np
import torch
import torch.nn as nn

# === EMBEDDINGS (from Phase 1) ===
words = ["the", "cat", "sat", "on", "mat"]
special_tokens = ['<PAD>', '<UNK>']
vocab = words + special_tokens
vocab_size = len(vocab)
d_model = 8
seq_len = 6

word_to_id = {word: i for i, word in enumerate(vocab)}

def tokenizer(sentence, word_to_id):
    words = sentence.lower().split()
    unk_id = word_to_id["<UNK>"]
    return [word_to_id.get(word, unk_id) for word in words]

sentence = "the cat sat on the mat"
tokens = tokenizer(sentence, word_to_id)

# Create embeddings
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
token_tensor = torch.tensor(tokens)
embeddings = embedding_layer(token_tensor)

print("✓ Embeddings created")
print("Embeddings shape:", embeddings.shape)
print("Original embeddings[0] (token 'the'):", embeddings[0])

# === POSITIONAL ENCODING ===
def get_positional_encoding_fast(max_seq_len, d_model):
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(0, d_model, 2)[np.newaxis, :]
    angles = positions / (10000 ** (dims / d_model))
    PE = np.zeros((max_seq_len, d_model))
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles)
    return PE

pe = get_positional_encoding_fast(seq_len, d_model)
pe_tensor = torch.tensor(pe, dtype=torch.float32)

print("\n✓ Positional encoding created")
print("PE shape:", pe_tensor.shape)
print("PE[0] (position 0):", pe_tensor[0])

# === ADD ===
embeddings_with_pos = embeddings + pe_tensor

print("\n✓ Addition complete!")
print("Final shape (unchanged!):", embeddings_with_pos.shape)
print("Embeddings with pos[0]:", embeddings_with_pos[0])
print("\nKey insight: Shape same (6,8), but values changed → word meaning + position!")
