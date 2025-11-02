from math import sqrt
import torch.nn as nn
import matplotlib.pyplot as plt



class AE(nn.Module):
    def __init__(self, n_features):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_features, round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),
        )
        self.dec = nn.Sequential(
            nn.Linear(round(sqrt(n_features)) + 1, round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), n_features),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode, 
