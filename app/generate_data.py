import numpy as np
import pandas as pd
import random
from utils import create_sequences # Import create_sequences from utils.py
import os
from sklearn.preprocessing import LabelEncoder

import numpy as np

def generate_dummy_sequence(seq_len=10):
    print(f"New random sequence of length {seq_len} generated.")
    sequence = []

    # Select one med per sequence (for realism)
    med_names = ['MedA', 'MedB', 'MedC', 'MedD', 'MedE']
    med_name = np.random.choice(med_names)
    dosesPerDay = np.random.randint(1, 5)
    servoNum = np.random.randint(1, 11)
    pills = np.random.randint(1, 6)  # dose per take
    pillsInTape = np.random.randint(30, 100)  # starting inventory

    base_timestamp = 1_700_000_000  # Base time for the sequence
    lastNotifyTime = 0  # Will be updated if low inventory is detected

    # Define ideal dose hours (equally spaced in a 24-hour day)
    interval = 24 / (dosesPerDay - 1) if dosesPerDay > 1 else 24
    scheduled_hours = [(8 + round(i * interval)) % 24 for i in range(dosesPerDay)]

    for i in range(seq_len):
        # Simulate time of entry
        hour = (8 + i * (24 // seq_len)) % 24
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)

        # Calculate current timestamp for this step
        elapsed_seconds = i * (24 * 3600 // seq_len)
        current_time = base_timestamp + elapsed_seconds

        # Simulate pill consumption
        if np.random.rand() < 0.7:  # 70% chance a dose is taken
            pillsInTape = max(0, pillsInTape - pills)

        # Notification flag if inventory is low
        notifiedLowDays = pillsInTape < 20

        # Triggered logic (hybrid): dose time or critically low inventory
        near_dose_time = any(abs(hour - h) <= 1 for h in scheduled_hours)
        triggered = near_dose_time or pillsInTape < 10

        # Update lastNotifyTime if low inventory notification is triggered
        if notifiedLowDays:
            lastNotifyTime = current_time

        sequence.append([
            med_name,
            hour,
            minute,
            second,
            pills,
            servoNum,
            pillsInTape,
            dosesPerDay,
            triggered,
            notifiedLowDays,
            lastNotifyTime
        ])

    print(f"New random sequence of length {seq_len} and {med_name} generated.")
    return sequence





def generate_dataset(num_samples=1000, seq_len=10):
    X = []
    y = []
    raw_scores = []

    for _ in range(num_samples):
        sequence = generate_dummy_sequence(seq_len)

        # --------- STEP 1: Temporal weights (more importance to recent events) ---------
        weights = np.linspace(1, 2, seq_len)  # e.g., [1.0, 1.1, ..., 2.0]

        # --------- STEP 2: Risk from triggered and notified flags ---------
        triggered_weighted = sum(weights[i] * (1 if step[8] else 0) for i, step in enumerate(sequence))
        notified_weighted = sum(weights[i] * (1 if step[9] else 0) for i, step in enumerate(sequence))

        # --------- STEP 3: Average gap between timestamps ---------
        time_gaps = []
        for i in range(1, seq_len):
            prev_time = sequence[i - 1][1] * 3600 + sequence[i - 1][2] * 60 + sequence[i - 1][3]
            curr_time = sequence[i][1] * 3600 + sequence[i][2] * 60 + sequence[i][3]
            time_gaps.append(abs(curr_time - prev_time))
        avg_time_gap = np.mean(time_gaps)

        # --------- STEP 4: Final dose inventory (pillsInTape) ---------
        low_inventory_penalty = 1 if sequence[-1][6] < 10 else 0  # Low inventory at last timestep

        # --------- STEP 5: Total risk score ---------
        risk_score = (
            0.05 * triggered_weighted +
            0.05 * notified_weighted +
            0.03 * avg_time_gap +
            0.02 * low_inventory_penalty
        )

        raw_scores.append(risk_score)
        X.append(sequence)

    # --------- STEP 6: Normalize risk scores to [0, 1] ---------
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    y = [[(score - min_score) / (max_score - min_score)] for score in raw_scores]

    return np.array(X, dtype=object), np.array(y)


import os
import numpy as np
import pandas as pd
from utils import create_sequences  # make sure utils.py is in the same directory or properly imported

def save_generated_data(filepath='dataset/train.csv', num_records=1000, seq_length=10):
    """
    Generates synthetic data using generate_dummy_sequence(), assigns risk scores,
    saves to CSV, and returns sequence/target arrays for model training.
    """
    from generate_data import generate_dummy_sequence  # Import your sequence generator
    X_all = []
    y_all = []
    raw_rows = []

    for _ in range(num_records):
        sequence = generate_dummy_sequence(seq_length)

        # Step 1: Temporal weights
        weights = np.linspace(1, 2, seq_length)

        # Step 2: Triggered + Notified weighted
        triggered_weighted = sum(weights[i] * (1 if step[8] else 0) for i, step in enumerate(sequence))
        notified_weighted = sum(weights[i] * (1 if step[9] else 0) for i, step in enumerate(sequence))

        # Step 3: Average time gap
        time_gaps = []
        for i in range(1, seq_length):
            prev_time = sequence[i - 1][1] * 3600 + sequence[i - 1][2] * 60 + sequence[i - 1][3]
            curr_time = sequence[i][1] * 3600 + sequence[i][2] * 60 + sequence[i][3]
            time_gaps.append(abs(curr_time - prev_time))
        avg_time_gap = np.mean(time_gaps)

        # Step 4: Low inventory
        low_inventory_penalty = 1 if sequence[-1][6] < 10 else 0

        # Step 5: Risk score
        raw_score = (
            0.05 * triggered_weighted +
            0.05 * notified_weighted +
            0.03 * avg_time_gap +
            0.02 * low_inventory_penalty
        )

        # Accumulate for normalization later
        X_all.append(sequence)
        y_all.append(raw_score)

    # Normalize risk scores to [0, 1]
    min_score, max_score = min(y_all), max(y_all)
    normalized_y = [(score - min_score) / (max_score - min_score) for score in y_all]

    # Flatten sequences and assign risk_score column
    flat_rows = []
    for i in range(len(X_all)):
        for step in X_all[i]:
            flat_rows.append(step + [normalized_y[i]])

    columns = [
        'med_name', 'hour', 'minute', 'second', 'pills', 'servoNum',
        'pillsInTape', 'dosesPerDay', 'triggered', 'notifiedLowDays',
        'lastNotifyTime', 'risk_score'
    ]

    raw_data_df = pd.DataFrame(flat_rows, columns=columns)

    # Ensure the 'dataset' directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_data_df.to_csv(filepath, index=False)
    print(f"Raw data saved to {filepath} with {len(X_all)} sequences.")

    # Encode med_name
    raw_data_df['med_name'] = raw_data_df['med_name'].astype('category').cat.codes

    # Now create sequences and targets
    sequences, targets = create_sequences(
        raw_data_df, seq_length=seq_length, input_dim=len(columns) - 1, target_col='risk_score'
    )
    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Generated targets shape: {targets.shape}")
    return sequences, targets


if __name__ == "__main__":
    dataset_dir = '../dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    print("Generating and saving training data...")
    X_train, y_train = save_generated_data(os.path.join(dataset_dir, 'train.csv'), num_records=8000, seq_length=10)
    np.save(os.path.join(dataset_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(dataset_dir, 'y_train.npy'), y_train)

    print("Generating and saving validation data...")
    X_val, y_val = save_generated_data(os.path.join(dataset_dir, 'val.csv'), num_records=1000, seq_length=10)
    np.save(os.path.join(dataset_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(dataset_dir, 'y_val.npy'), y_val)

    print("Generating and saving test data...")
    X_test, y_test = save_generated_data(os.path.join(dataset_dir, 'test.csv'), num_records=2000, seq_length=10)
    np.save(os.path.join(dataset_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(dataset_dir, 'y_test.npy'), y_test)

    print("All datasets generated and saved successfully.")
