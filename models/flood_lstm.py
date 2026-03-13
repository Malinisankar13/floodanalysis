import torch
import torch.nn as nn

class FloodLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        # We want to predict the area at t+1 based on t steps
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the hidden state of the last layer
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out
