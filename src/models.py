import torch
import torch.nn as nn


class DFN(nn.Module):
    def __init__(self, layer_dims, activation=nn.ReLU):
        super(DFN, self).__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i == len(layer_dims) - 1:
                break
            layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(-1, self.batch_size, len(input)))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return y_pred.view(-1)
