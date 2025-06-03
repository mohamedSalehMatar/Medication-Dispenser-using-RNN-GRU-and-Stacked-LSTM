import numpy as np
import tensorflow as tf
from keras import layers, models

def build_adherence_rnn(input_dim=9, hidden_dim=32, output_dim=1):
    model = models.Sequential([
        layers.Input(shape=(None, input_dim)),
        layers.LSTM(hidden_dim, return_sequences=False),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    return model

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
            np.random.randint(0, 5),              # missed doses last 24h
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
    return np.array(X), np.array(y)

# Example usage
if __name__ == "__main__":
    model = build_adherence_rnn()
    model.summary()
    X, y = generate_dataset(32, 10)
    output = model(X)
    print("Output shape:", output.shape)
    print("Output:", output)
