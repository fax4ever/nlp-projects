import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Sample dataset (text, label)
dataset = [
    ("The history of ancient art is fascinating", "Art"),
    ("Quantum computing is the future of science", "Science"),
    ("Business trends and global markets", "Business"),
    ("Literature and music reflect cultural values", "Art"),
    ("New discoveries in biology and physics", "Science"),
    ("Market analysis and finance predictions", "Business")
]

# 2. Tokenize and build vocab
def tokenize(text):
    return text.lower().split()

all_tokens = [token for text, _ in dataset for token in tokenize(text)]
vocab = {word: idx+1 for idx, word in enumerate(set(all_tokens))}  # idx+1 to reserve 0 for padding
vocab["<PAD>"] = 0
vocab_size = len(vocab)

# Encode texts as sequences of token indices
def encode(text):
    return [vocab[token] for token in tokenize(text)]

# Encode labels
labels = [label for _, label in dataset]
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Prepare data tensors
encoded_texts = [encode(text) for text, _ in dataset]
offsets = [0] + [len(seq) for seq in encoded_texts[:-1]]
offsets = torch.tensor(offsets).cumsum(dim=0)
input_tensor = torch.tensor([token for seq in encoded_texts for token in seq])
offsets_tensor = offsets
labels_tensor = torch.tensor(label_ids)

# Train/test split
train_idx, test_idx = train_test_split(list(range(len(labels))), test_size=0.3, random_state=42)
train_input = input_tensor[torch.cat([torch.arange(offsets[i], offsets[i] + len(encoded_texts[i])) for i in train_idx])]
train_offsets = torch.tensor([0] + [len(encoded_texts[i]) for i in train_idx[:-1]]).cumsum(dim=0)
train_labels = labels_tensor[train_idx]

test_input = input_tensor[torch.cat([torch.arange(offsets[i], offsets[i] + len(encoded_texts[i])) for i in test_idx])]
test_offsets = torch.tensor([0] + [len(encoded_texts[i]) for i in test_idx[:-1]]).cumsum(dim=0)
test_labels = labels_tensor[test_idx]

# 3. Define Model with Embedding + MLP
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Model setup
embed_dim = 32
model = TextClassifier(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)

# 4. Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 50

for epoch in range(epochs):
    model.train()
    output = model(train_input, train_offsets)
    loss = criterion(output, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    predictions = model(test_input, test_offsets)
    predicted_labels = predictions.argmax(dim=1)
    accuracy = (predicted_labels == test_labels).float().mean()
    print(f"\nTest Accuracy: {accuracy.item():.2f}")
    print("Predictions:", label_encoder.inverse_transform(predicted_labels.numpy()))
    print("Ground Truth:", label_encoder.inverse_transform(test_labels.numpy()))
