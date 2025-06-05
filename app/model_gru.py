import numpy as np
import tensorflow as tf
from keras import layers, models

class GRUModel:
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=1, dropout=0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(None, self.input_dim)),
            layers.GRU(self.hidden_dim, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(self.output_dim, activation='sigmoid')
        ])
        return model

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model

# Dummy data generator (same as original)

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

if __name__ == "__main__":
    model_obj = GRUModel()
    model_obj.summary()
    X, y = generate_dataset(32, 10)
    output = model_obj.get_model()(X)
    print("Output shape:", output.shape)
    print("Output:", output)
