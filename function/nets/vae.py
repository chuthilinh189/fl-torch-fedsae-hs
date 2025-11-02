from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import torch


class VariationalEncoder(nn.Module):
    def __init__(self, n_features):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, round(n_features * 0.5))
        self.linear2 = nn.Linear(
            round(n_features * 0.5), round(sqrt(n_features)) + 1)
        self.linear3 = nn.Linear(
            round(n_features * 0.5), round(sqrt(n_features)) + 1)
        
        
        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # .cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()  # .cuda()

        # Phát hiện thiết bị (GPU hoặc CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Chuyển sampling distribution sang thiết bị tương ứng
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)
        

        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, n_features):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(
            round(sqrt(n_features)) + 1, round(n_features * 0.5))
        self.linear2 = nn.Linear(round(n_features * 0.5), n_features)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        return self.linear2(z)


class VAE(nn.Module):
    def __init__(self, n_features=0):
        super(VAE, self).__init__()
        self.enc = VariationalEncoder(n_features)
        self.dec = Decoder(n_features)

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode
