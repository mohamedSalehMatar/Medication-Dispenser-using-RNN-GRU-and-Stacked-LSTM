import numpy as np
import pandas as pd

def create_sequences(data_df, seq_length=3, input_dim=5):
    """
    Creates sequences of `seq_length` from the input DataFrame.
    Each sequence is associated with a binary target: whether the next dose is missed (0) or taken (1).
    """
    sequences = []
    targets = []

    if len(data_df) < seq_length + 1:
        print(f"Warning: Not enough data ({len(data_df)} records) to form sequence+target of length {seq_length+1}.")
        return np.array([]).reshape(0, seq_length, input_dim), np.array([]).reshape(0, 1)

    for i in range(len(data_df) - seq_length):
        seq_slice = data_df.iloc[i : i + seq_length]
        next_status = data_df.iloc[i + seq_length]['status']  # 0 or 1

        X = seq_slice[['medicine', 'dose', 'scheduled_time', 'confirmation_time', 'status']].values.astype(np.float32)
        y = [float(next_status)]

        sequences.append(X)
        targets.append(y)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def load_data_from_csv(file_path):
    """
    Load a CSV file and clean it.
    Assumes all features are numerical.
    """
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading CSV from {file_path}: {e}")
        return pd.DataFrame()


# Optional: quick test
if __name__ == "__main__":
    print("Testing create_sequences on dummy medicine data...")

    dummy_df = pd.DataFrame({
        'medicine': np.random.randint(0, 5, 10),
        'dose': np.random.randint(1, 4, 10),
        'scheduled_time': np.random.randint(0, 86400, 10),
        'confirmation_time': np.random.randint(0, 86400, 10),
        'status': np.random.randint(0, 2, 10),
    })

    X, y = create_sequences(dummy_df, seq_length=3)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Example sequence:\n", X[0])
    print("Target (will user take next dose?):", y[0])
