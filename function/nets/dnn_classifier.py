import torch
import torch.nn as nn

class DNNClassifier(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super(DNNClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, n_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def calculate_loss(self, x, y):
        logits = self.forward(x)
        return self.loss_fn(logits, y)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
