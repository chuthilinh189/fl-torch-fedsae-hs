import torch
import torch.nn as nn
from math import sqrt


class FedDetect(nn.Module):
    """Encoder–decoder–predictor; loss is defined externally by clients."""

    def __init__(self, n_features, n_classes=2):
        super(FedDetect, self).__init__()

        bottleneck = round(sqrt(n_features)) + 1

        self.encoder = nn.Sequential(
            nn.Linear(n_features, round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), n_features),
        )

        self.predictor = nn.Sequential(
            nn.Linear(bottleneck, max(4, round(bottleneck * 0.75))),
            nn.ReLU(),
            nn.Linear(max(4, round(bottleneck * 0.75)), max(2, round(bottleneck * 0.5))),
            nn.ReLU(),
            nn.Linear(max(2, round(bottleneck * 0.5)), n_classes),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        logits = self.predictor(latent.view(latent.size(0), -1))
        return logits, recon

    def predict(self, x):
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.argmax(logits, dim=1)
