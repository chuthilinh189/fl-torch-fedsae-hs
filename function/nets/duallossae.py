import torch
import torch.nn as nn
from math import sqrt

class DualLossAE(nn.Module):
    def __init__(self, n_features, epsilon=1e-5):
        super(DualLossAE, self).__init__()
        self.epsilon = epsilon
        
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
        self.loss_function = nn.MSELoss()
    
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode
    
    def calculate_loss(self, x, y):
        _, decode = self.forward(x)
        re_loss = self.loss_function(decode, x)
        dual_loss = (1 - y) * re_loss + y * 0.001 / (re_loss + self.epsilon)
        return dual_loss.mean()
