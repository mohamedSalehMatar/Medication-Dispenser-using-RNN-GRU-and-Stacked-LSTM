import numpy as np
import pandas as pd
import random

def create_sequences(data_df, seq_length=10, input_dim=9):
    """
    Creates sequences (X) and targets (y) from a flat DataFrame.
    It also calculates a pseudo-risk score for each sequence.

    Args:
        data_df (pd.DataFrame): DataFrame containing the raw data.
                                Expected columns match generate_dummy_record output.
        seq_length (int): The length of each sequence.
        input_dim (int): The number of features in each time step.

    Returns:
        tuple: (X_sequences, y_targets) where X_sequences is a numpy array of sequences
               and y_targets is a numpy array of corresponding risk scores.
    """
    sequences = []
    targets = []
    raw_scores = []

    # Ensure data is long enough to create at least one sequence
    if len(data_df) < seq_length:
        print(f"Warning: Not enough data ({len(data_df)} records) to form a sequence of length {seq_length}.")
        return np.array([]).reshape(0, seq_length, input_dim), np.array([]).reshape(0, 1)

    for i in range(len(data_df) - seq_length + 1):
        sequence_data = data_df.iloc[i : i + seq_length].values

        # Calculate risk score for this sequence
        # This logic is adapted from the original generate_dataset
        risk_score = 0
        risk_score += (sequence_data[:, 1].mean() / 1200) * 25   # delay_seconds normalized and weighted
        risk_score += (sequence_data[:, 5].mean() / 720) * 20   # time_since_last_dose normalized and weighted
        risk_score += (sequence_data[:, 2].mean()) * 10         # confirmed (0 or 1) weighted
        risk_score += (sequence_data[:, 6].mean() / 15) * 30    # missed_doses_24h normalized and weighted
        risk_score += (sequence_data[:, 7].mean()) * 10         # was_prev_dose_late weighted

        sequences.append(sequence_data)
        raw_scores.append(risk_score)

    # Min-max normalize risk scores to 0-1 range
    if raw_scores:
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score == min_score: # Avoid division by zero if all scores are same
            targets = [[0.5] for _ in raw_scores] # Assign a neutral score
        else:
            targets = [[(score - min_score) / (max_score - min_score)] for score in raw_scores]
    else:
        targets = []


    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# This function might not be used if generate_data.py handles direct saving
# but is included for completeness based on previous discussions.
def load_data_from_csv(filepath):
    """Loads data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage of create_sequences
    print("Testing create_sequences in utils.py")
    # Generate some dummy flat data for testing
    num_dummy_records = 50
    dummy_data = []
    for _ in range(num_dummy_records):
        scheduled_time = np.random.randint(0, 1440)
        delay_seconds = np.random.randint(0, 600)
        confirmed = np.random.randint(0, 2)
        day_of_week = np.random.randint(0, 7)
        hour_of_day = np.random.randint(0, 24)
        time_since_last_dose = np.random.randint(0, 720)
        missed_doses_24h = np.random.randint(0, 5)
        was_prev_dose_late = np.random.randint(0, 2)
        first_dose_of_day = np.random.randint(0, 2)
        dummy_data.append([
            scheduled_time, delay_seconds, confirmed, day_of_week,
            hour_of_day, time_since_last_dose, missed_doses_24h,
            was_prev_dose_late, first_dose_of_day
        ])
    dummy_df = pd.DataFrame(dummy_data, columns=[
        'scheduled_time', 'delay_seconds', 'confirmed', 'day_of_week',
        'hour_of_day', 'time_since_last_dose', 'missed_doses_24h',
        'was_prev_dose_late', 'first_dose_of_day'
    ])

    X_seq, y_target = create_sequences(dummy_df, seq_length=10)
    print(f"Generated X_seq shape: {X_seq.shape}")
    print(f"Generated y_target shape: {y_target.shape}")
    if X_seq.shape[0] > 0:
        print("First sequence X:\n", X_seq[0])
        print("First target y:", y_target[0])