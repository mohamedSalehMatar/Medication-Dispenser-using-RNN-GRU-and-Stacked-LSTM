import torch
import torch.nn as nn

class AdherenceRNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, num_layers=1, output_dim=1):
        super(AdherenceRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)     # out: (batch_size, sequence_length, hidden_dim)
        out = out[:, -1, :]       # take output from the last time step
        out = self.fc(out)        # out: (batch_size, output_dim)
        return self.sigmoid(out)  # binary classification output

# Example usage
if __name__ == "__main__":
    model = AdherenceRNN()
    dummy_input = torch.randn(4, 10, 9)   # batch of 4, sequence length 10, 9 features
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be (4, 1)
    print("Output:", output)
