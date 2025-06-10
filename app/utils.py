import numpy as np
import pandas as pd
import random

def create_sequences(data_df, seq_length=10, input_dim=11):
    
    sequences = []
    targets = []
    raw_scores = []

    # Encode med_name column to numeric labels
    med_name_unique = data_df['med_name'].unique()
    med_name_to_num = {name: idx for idx, name in enumerate(med_name_unique)}

    # Ensure data is long enough to create at least one sequence
    if len(data_df) < seq_length:
        print(f"Warning: Not enough data ({len(data_df)} records) to form a sequence of length {seq_length}.")
        return np.array([]).reshape(0, seq_length, input_dim), np.array([]).reshape(0, 1)

    for i in range(len(data_df) - seq_length + 1):
        sequence_data = data_df.iloc[i : i + seq_length].copy()

        # Replace med_name strings with numeric labels
        print(f"Processing sequence from index {i} to {i + seq_length - 1}")
        sequence_data['med_name'] = sequence_data['med_name'].map(med_name_to_num)

        sequence_data_values = sequence_data.values.astype(np.float32)

        # Calculate risk score for this sequence
        # Adapted to new features:
        # med_name (encoded numeric) at index 0
        # hour=1, minute=2, second=3, pills=4, servoNum=5, pillsInTape=6, dosesPerDay=7, triggered=8, notifiedLowDays=9, lastNotifyTime=10
        risk_score = 0
        risk_score += (sequence_data_values[:, 0].mean()) * 5          # med_name encoded weighted
        risk_score += (sequence_data_values[:, 1].mean() / 24) * 10    # hour normalized and weighted
        risk_score += (sequence_data_values[:, 4].mean() / 5) * 20     # pills normalized and weighted
        risk_score += (sequence_data_values[:, 8].mean()) * 15         # triggered weighted
        risk_score += (sequence_data_values[:, 9].mean()) * 10         # notifiedLowDays weighted

        sequences.append(sequence_data_values)
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

if __name__ == "__main__":
    # Example usage of create_sequences
    print("Testing create_sequences in utils.py")
    # Generate some dummy flat data for testing
    num_dummy_records = 50
    dummy_data = []
    med_names = ['MedA', 'MedB', 'MedC', 'MedD', 'MedE']
    for _ in range(num_dummy_records):
        med_name = np.random.choice(med_names)
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        pills = np.random.randint(1, 6)
        servoNum = np.random.randint(1, 11)
        pillsInTape = np.random.randint(0, 101)
        dosesPerDay = np.random.randint(1, 5)
        triggered = np.random.choice([True, False])
        notifiedLowDays = np.random.choice([True, False])
        lastNotifyTime = np.random.randint(1_600_000_000, 1_700_000_000)
        dummy_data.append([
            med_name, hour, minute, second, pills, servoNum,
            pillsInTape, dosesPerDay, triggered, notifiedLowDays, lastNotifyTime
        ])
    dummy_df = pd.DataFrame(dummy_data, columns=[
        'med_name', 'hour', 'minute', 'second', 'pills', 'servoNum',
        'pillsInTape', 'dosesPerDay', 'triggered', 'notifiedLowDays', 'lastNotifyTime'
    ])

    X_seq, y_target = create_sequences(dummy_df, seq_length=10, input_dim=11)
    print(f"Generated X_seq shape: {X_seq.shape}")
    print(f"Generated y_target shape: {y_target.shape}")
    if X_seq.shape[0] > 0:
        print("First sequence X:\n", X_seq[0])
        print("First target y:", y_target[0])
