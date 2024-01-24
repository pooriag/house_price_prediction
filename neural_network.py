import torch
from torch import nn

class house_price_network(nn.Module):
    def __init__(self, features):
        super(house_price_network, self).__init__()

        self.features = features
        self.losses = []

        self.layer = nn.Linear(features, 1)

    def forward(self, x):
        return self.layer(x)

    def predic(self, x):
        X = torch.Tensor(x)
        x_vertical = X.resize(1, 1, )
        print(x_vertical)