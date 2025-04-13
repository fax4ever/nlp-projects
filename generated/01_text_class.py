import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Define the keyword-based vocabulary
vocabulary = ["art", "culture", "history", "science", "literature", "music"]
vocab_size = len(vocabulary)

# 2. Sample dataset (text, label)
dataset = [
    ("The history of ancient art is fascinating", 1),
    ("Latest developments in quantum computing", 0),
    ("Literature and music reflect cultural values", 1),
    ("Data science is revolutionizing industries", 0),
    ("Art and culture play key roles in society", 1),
    ("Business trends and global markets", 0)
]

# 3. Text to binary vector based on vocabulary
def text_to_vector(text, vocabulary):
    text = text.lower()
    return np.array([1 if word in text else 0 for word in vocabulary], dtype=np.float32)

# 4. Prepare data
X = np.array([text_to_vector(text, vocabulary) for text, _ in dataset])
y = np.array([label for _, label in dataset], dtype=np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)  # (batch_size, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

# 5. Define MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLPClassifier(input_size=vocab_size)

# 6. Train the model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 7. Evaluate on test set
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy.item():.2f}")
