"""Vocabulary Creation"""

words = ["the", "cat", "sat", "on", "mat"]
special_tokens = ['<PAD>', '<UNK>']

vocab = words + special_tokens

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for i, word in enumerate(vocab)}


print("Vocabulary Size:", len(vocab))
print("Word to ID map:", word_to_id)



"""Tokenization"""

def tokenizer(sentence, word_to_id):
    words = sentence.lower().split()
    
    unk_id = word_to_id["<UNK>"]    #   # Look up each word's ID. If not found, use the ID for <UNK>
    token_ids = [word_to_id.get(word, unk_id) for word in words]
    
    return token_ids

sentence = "the cat sat on the mat"
tokens = tokenizer(sentence, word_to_id)

print(f"Sentence: '{sentence}'")
print(f"Token IDs: {tokens}")