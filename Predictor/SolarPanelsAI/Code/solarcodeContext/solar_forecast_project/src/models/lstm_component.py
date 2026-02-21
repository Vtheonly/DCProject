
import torch
import torch.nn as nn
import numpy as np

# Residual Correction LSTM
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2) # Outputs: [Correction, Uncertainty]

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class HeteroscedasticLoss(nn.Module):
    def forward(self, pred, target):
        delta = pred[:, 0]
        log_var = pred[:, 1]
        residual = target.squeeze()
        precision = torch.exp(-log_var)
        loss = 0.5 * precision * (residual - delta)**2 + 0.5 * log_var
        return loss.mean()
