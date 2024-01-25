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

    def predict(self, x):
        X = torch.Tensor(x)
        return self.layer(X)

    def train(self, x, lable):
        Lable = torch.Tensor(lable)
        loss_function = nn.MSELoss()
        learning_rate = (1e-2)
        epochs = 10000

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for i in range(epochs):
            y_predict = self.predict(x)
            loss = loss_function(y_predict.squeeze(), Lable)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            self.losses.append(loss.item())