# Download GloVe (if not done yet)
# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip

import numpy as np

def load_glove_embeddings(path, vocab, embed_dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim)).astype(np.float32)
    embeddings[vocab["<PAD>"]] = np.zeros(embed_dim)  # padding gets zero vector

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split()
            word, vector = tokens[0], tokens[1:]
            if word in vocab:
                embeddings[vocab[word]] = np.array(vector, dtype=np.float32)
    return torch.tensor(embeddings)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pretrained_embeddings=None):
        super(TextClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Optional: freeze embeddings
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
