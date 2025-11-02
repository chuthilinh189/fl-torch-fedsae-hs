import torch
import torch.nn as nn
from math import sqrt

class SupAE(nn.Module):
    def __init__(self, n_features):
        super(SupAE, self).__init__()
        
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
        self.latent_distance = nn.L1Loss()
    
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode
    
    def calculate_loss(self, x, y):
        z, decode = self.forward(x)
        re_loss = self.loss_function(decode, x)
        # Xử lý y để khớp với z
        if y.dim() == 1:
            y = y.view(-1, 1)
        elif y.shape[1] != 1:
            y = y.mean(dim=1, keepdim=True)
        
        y_expanded = y.expand_as(z)  # đảm bảo y có cùng shape với z

        latent_loss = self.latent_distance(z, y_expanded)
        return re_loss, latent_loss
