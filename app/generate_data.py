import numpy as np
import pandas as pd
import random
from utils import create_sequences # Import create_sequences from utils.py
import os

def generate_dummy_sequence(seq_len=10):
    sequence = []
    med_names = ['MedA', 'MedB', 'MedC', 'MedD', 'MedE']
    for _ in range(seq_len):
        med_name = np.random.choice(med_names)  # string
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        pills = np.random.randint(1, 6)  # dose by take
        servoNum = np.random.randint(1, 11)  # drawer number
        pillsInTape = np.random.randint(0, 101)  # remaining pills
        dosesPerDay = np.random.randint(1, 5)
        triggered = np.random.choice([True, False])
        notifiedLowDays = np.random.choice([True, False])
        lastNotifyTime = np.random.randint(1_600_000_000, 1_700_000_000)  # example timestamp range

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
            lastNotifyTime,
        ])
    return sequence

def generate_dataset(num_samples=1000, seq_len=10):
    X = []
    y = []
    raw_scores = []
    for _ in range(num_samples):
        sequence = generate_dummy_sequence(seq_len)
        # Better pseudo logic incorporating all features
        risk_score = 0
        # Since med_name is string, skip it in risk score calculation
        # Use numeric features indices accordingly:
        # hour=1, minute=2, second=3, pills=4, servoNum=5, pillsInTape=6, dosesPerDay=7, triggered=8, notifiedLowDays=9, lastNotifyTime=10
        risk_score += (np.mean([item[1] for item in sequence]) / 24) * 10  # hour normalized and weighted
        risk_score += (np.mean([item[4] for item in sequence]) / 5) * 20   # pills normalized and weighted
        risk_score += (np.mean([1 if item[8] else 0 for item in sequence])) * 15  # triggered weighted
        risk_score += (np.mean([1 if item[9] else 0 for item in sequence])) * 10  # notifiedLowDays weighted
        raw_scores.append(risk_score)
        X.append(sequence)
    # Min-max normalize risk scores to 0-1 range
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    y = [ [(score - min_score) / (max_score - min_score)] for score in raw_scores ]
    return np.array(X, dtype=object), np.array(y)

def save_generated_data(filepath='dataset/train.csv', num_records=1000, seq_length=10):
    """
    Generates raw data, saves it to CSV, and then creates sequences and targets.
    This function now uses create_sequences from utils.py.
    """
    raw_data = []
    for _ in range(num_records):
        raw_data.extend(generate_dummy_sequence(seq_length))
    columns = [
        'med_name', 'hour', 'minute', 'second', 'pills', 'servoNum',
        'pillsInTape', 'dosesPerDay', 'triggered', 'notifiedLowDays', 'lastNotifyTime'
    ]
    raw_data_df = pd.DataFrame(raw_data, columns=columns)

    # Ensure the 'dataset' directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_data_df.to_csv(filepath, index=False)
    print(f"Raw data saved to {filepath} with {num_records} sequences.")

    # Now create sequences and targets using the utility function
    sequences, targets = create_sequences(raw_data_df, seq_length=seq_length, input_dim=len(columns))
    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Generated targets shape: {targets.shape}")
    return sequences, targets

if __name__ == "__main__":
    dataset_dir = 'dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    print("Generating and saving training data...")
    X_train, y_train = save_generated_data(os.path.join(dataset_dir, 'train.csv'), num_records=80000, seq_length=100)
    np.save(os.path.join(dataset_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(dataset_dir, 'y_train.npy'), y_train)

    print("Generating and saving validation data...")
    X_val, y_val = save_generated_data(os.path.join(dataset_dir, 'val.csv'), num_records=10000, seq_length=100)
    np.save(os.path.join(dataset_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(dataset_dir, 'y_val.npy'), y_val)

    print("Generating and saving test data...")
    X_test, y_test = save_generated_data(os.path.join(dataset_dir, 'test.csv'), num_records=20000, seq_length=100)
    np.save(os.path.join(dataset_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(dataset_dir, 'y_test.npy'), y_test)

    print("All datasets generated and saved successfully.")
