import torch
import torch.nn as nn
import numpy as np

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
        out, _ = self.lstm(x)  # out: (batch_size, sequence_length, hidden_dim)
        out = out[:, -1, :]    # take output from the last time step
        out = self.fc(out)     # out: (batch_size, output_dim)
        return self.sigmoid(out)  # continuous risk score

# Dummy data generator

def generate_dummy_sequence(seq_len=10):
    return np.array([
        [
            np.random.randint(0, 1440),           # scheduled_time (minutes since midnight)
            np.random.randint(0, 600),            # delay_seconds
            np.random.randint(0, 2),              # confirmed
            np.random.randint(0, 7),              # day_of_week
            np.random.randint(0, 24),             # hour_of_day
            np.random.randint(0, 720),            # time_since_last_dose (minutes)
            np.random.randint(0, 10),              # missed doses last 24h
            np.random.randint(0, 2),              # was previous dose late
            np.random.randint(0, 2),              # first dose of day
        ]
        for _ in range(seq_len)
    ], dtype=np.float32)

def generate_dataset(num_samples=1000, seq_len=10):
    X = []
    y = []
    for _ in range(num_samples):
        sequence = generate_dummy_sequence(seq_len)
        risk_score = np.clip((sequence[:, 1].mean() / 600) + (sequence[:, 5].mean() / 720), 0, 1)  # pseudo logic
        X.append(sequence)
        y.append([risk_score])
    return torch.tensor(X), torch.tensor(y)

# Example usage
if __name__ == "__main__":
    model = AdherenceRNN()
    X, y = generate_dataset(32, 10)
    output = model(X)
    print("Output shape:", output.shape)  # should be (32, 1)
    print("Output:", output)
