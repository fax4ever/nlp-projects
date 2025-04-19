import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# === Dataset ===
class TextDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = X          # shape: [num_samples, max_len]
        self.y = y          # shape: [num_samples]
        self.lengths = lengths  # shape: [num_samples]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

# === Model from earlier ===
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        attn_weights = self.attention(lstm_out).squeeze(2)
        attn_weights = nn.functional.softmax(attn_weights, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        return self.classifier(context)

# === Training ===
def train_model(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            text, labels, lengths = batch
            text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)

            optimizer.zero_grad()
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        evaluate_model(model, val_loader, device)

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            text, labels, lengths = batch
            text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)

            predictions = model(text, lengths)
            predicted = torch.argmax(predictions, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Validation Accuracy: {acc:.2f}")

