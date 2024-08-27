import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMTradingAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMTradingAgent, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_size)
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def act(self, state):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Prepare state as a tensor
            state = torch.FloatTensor([state]).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            q_values = self.forward(state)
            action = torch.argmax(q_values, dim=1).item()
        self.train()  # Set the model back to training mode
        return action

    def store_outcome(self, reward):
        # Tutaj możesz dodać logikę zapisywania nagród dla agenta
        pass

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
