import numpy as np
import pandas as pd

def create_sequences(data_df, seq_length=10, input_dim=9):
    sequences = []
    targets = []

    if len(data_df) < seq_length:
        print(f"Warning: Not enough data ({len(data_df)} records) to form a sequence of length {seq_length}.")
        return np.array([]).reshape(0, seq_length, input_dim), np.array([]).reshape(0, 1)

    feature_cols = [col for col in data_df.columns if col != 'confirmed']

    for i in range(len(data_df) - seq_length + 1):
        sequence_data = data_df.iloc[i : i + seq_length]

        X = sequence_data[feature_cols].values.astype(np.float32)
        y = data_df.iloc[i + seq_length - 1]['confirmed']  # label of the last row in the sequence

        sequences.append(X)
        targets.append([float(y)])

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def load_data_from_csv(file_path):
    """Load CSV file into pandas DataFrame and drop any NaNs."""
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading CSV from {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if failed


# Optional: run test
if __name__ == "__main__":
    print("Testing create_sequences on dummy data...")

    dummy_df = pd.DataFrame({
        'scheduled_time': np.random.randint(0, 1500, 20),
        'delay_seconds': np.random.randint(0, 1000, 20),
        'day_of_week': np.random.randint(0, 7, 20),
        'hour_of_day': np.random.randint(0, 24, 20),
        'time_since_last_dose': np.random.randint(0, 700, 20),
        'missed_doses_24h': np.random.randint(0, 5, 20),
        'was_prev_dose_late': np.random.randint(0, 2, 20),
        'first_dose_of_day': np.random.randint(0, 2, 20),
        'confirmed': np.random.randint(0, 2, 20),
    })

    X, y = create_sequences(dummy_df, seq_length=5, input_dim=8)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
