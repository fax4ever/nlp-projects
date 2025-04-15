from torch import nn

class TextClassifier(nn.Module):
    def __init__(self, input_size):
        super(TextClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 3)
        )

    def forward(self, x):
        return self.model(x)