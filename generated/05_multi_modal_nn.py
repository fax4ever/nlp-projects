import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 text_embed_dim,
                 category_vocab_size,
                 common_dim=64,
                 num_numeric_features=4,
                 output_classes=3):

        super(MultiModalNN, self).__init__()

        # Text input
        self.text_embedding = nn.EmbeddingBag(vocab_size, text_embed_dim, mode='mean')
        self.text_proj = nn.Linear(text_embed_dim, common_dim)

        # Categorical input
        self.category_embedding = nn.Embedding(category_vocab_size, 32)
        self.category_proj = nn.Linear(32, common_dim)

        # Numeric input
        self.numeric_proj = nn.Linear(num_numeric_features, common_dim)

        # Combined feedforward classifier
        self.classifier = nn.Sequential(
            nn.Linear(common_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_classes)
        )

    def forward(self, text_input, text_offsets, category_input, numeric_input):
        # text_input: [B * L], text_offsets: [B], category_input: [B], numeric_input: [B x F]
        text_feat = self.text_embedding(text_input, text_offsets)       # [B x text_embed_dim]
        text_feat = self.text_proj(text_feat)                           # [B x 64]

        cat_feat = self.category_embedding(category_input)              # [B x 32]
        cat_feat = self.category_proj(cat_feat)                         # [B x 64]

        num_feat = self.numeric_proj(numeric_input)                     # [B x 64]

        combined = torch.cat([text_feat, cat_feat, num_feat], dim=1)   # [B x 192]
        output = self.classifier(combined)
        return output

batch_size = 4
vocab_size = 1000
category_vocab_size = 20
text_embed_dim = 50
num_numeric_features = 4
output_classes = 3

model = MultiModalNN(vocab_size, text_embed_dim, category_vocab_size,
                     common_dim=64, num_numeric_features=num_numeric_features,
                     output_classes=output_classes)

# Fake inputs
text_input = torch.randint(0, vocab_size, (20,))        # Flattened text tokens from 4 samples
text_offsets = torch.tensor([0, 5, 10, 15])              # Beginning of each sample in flattened tensor
category_input = torch.randint(0, category_vocab_size, (4,))
numeric_input = torch.randn(4, num_numeric_features)

output = model(text_input, text_offsets, category_input, numeric_input)
print(output.shape)  # [4 x output_classes]
