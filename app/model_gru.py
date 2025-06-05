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

import random

def generate_dummy_sequence(seq_len=10):
    sequence = []
    for _ in range(seq_len):
        scheduled_time = np.random.randint(0, 1440)           # minutes since midnight
        delay_seconds = np.random.randint(0, 600)
        confirmed = np.random.randint(0, 2)
        day_of_week = np.random.randint(0, 7)
        hour_of_day = np.random.randint(0, 24)
        time_since_last_dose = np.random.randint(0, 720)
        missed_doses_24h = np.random.randint(0, 5)
        was_prev_dose_late = np.random.randint(0, 2)
        first_dose_of_day = np.random.randint(0, 2)

        # Introduce slight anomalies/outliers randomly
        if random.random() < 0.05:
            delay_seconds += np.random.randint(600, 1200)  # large delay anomaly
        if random.random() < 0.03:
            missed_doses_24h += np.random.randint(5, 10)  # missed doses anomaly

        sequence.append([
            scheduled_time,
            delay_seconds,
            confirmed,
            day_of_week,
            hour_of_day,
            time_since_last_dose,
            missed_doses_24h,
            was_prev_dose_late,
            first_dose_of_day,
        ])
    return np.array(sequence, dtype=np.float32)

def generate_dataset(num_samples=1000, seq_len=10):
    X = []
    y = []
    raw_scores = []
    for _ in range(num_samples):
        sequence = generate_dummy_sequence(seq_len)
        # Better pseudo logic incorporating all features
        risk_score = 0
        risk_score += (sequence[:, 1].mean() / 1200) * 25  # delay_seconds normalized and weighted
        risk_score += (sequence[:, 5].mean() / 720) * 20   # time_since_last_dose normalized and weighted
        risk_score += (sequence[:, 2].mean()) * 10         # confirmed (0 or 1) weighted
        risk_score += (sequence[:, 6].mean() / 15) * 30    # missed_doses_24h normalized and weighted
        risk_score += (sequence[:, 7].mean()) * 10         # was_prev_dose_late weighted
        raw_scores.append(risk_score)
        X.append(sequence)
    # Min-max normalize risk scores to 0-1 range
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    y = [ [(score - min_score) / (max_score - min_score)] for score in raw_scores ]
    return np.array(X), np.array(y)

if __name__ == "__main__":
    model_obj = GRUModel()
    model_obj.summary()
    X, y = generate_dataset(32, 10)
    output = model_obj.get_model()(X)
    print("Output shape:", output.shape)
    print("Output:", output)
