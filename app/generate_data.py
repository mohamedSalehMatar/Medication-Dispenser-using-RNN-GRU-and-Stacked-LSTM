import numpy as np
import pandas as pd
import random
from utils import create_sequences # Import create_sequences from utils.py

def generate_dummy_record():
    """Generates a single dummy record (not a sequence)."""
    scheduled_time = np.random.randint(0, 1440)
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
        delay_seconds += np.random.randint(600, 1200) # large delay anomaly
    if random.random() < 0.03:
        missed_doses_24h += np.random.randint(5, 10) # missed doses anomaly

    return [
        scheduled_time, delay_seconds, confirmed, day_of_week,
        hour_of_day, time_since_last_dose, missed_doses_24h,
        was_prev_dose_late, first_dose_of_day
    ]

def generate_raw_dataset(num_records=1000):
    """Generates a flat dataset of raw records."""
    data = []
    for _ in range(num_records):
        data.append(generate_dummy_record())
    
    columns = [
        'scheduled_time', 'delay_seconds', 'confirmed', 'day_of_week',
        'hour_of_day', 'time_since_last_dose', 'missed_doses_24h',
        'was_prev_dose_late', 'first_dose_of_day'
    ]
    return pd.DataFrame(data, columns=columns)

def save_generated_data(filepath='data/train.csv', num_records=1000, seq_length=10):
    """
    Generates raw data, saves it to CSV, and then creates sequences and targets.
    This function now uses create_sequences from utils.py.
    """
    raw_data_df = generate_raw_dataset(num_records)
    
    # Ensure the 'data' directory exists
    import os
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_data_df.to_csv(filepath, index=False)
    print(f"Raw data saved to {filepath} with {num_records} records.")

    # Now create sequences and targets using the utility function
    # Note: If you want to save X and Y separately, you'd do that here.
    # For this example, we'll just demonstrate their creation.
    sequences, targets = create_sequences(raw_data_df, seq_length=seq_length, input_dim=9)
    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Generated targets shape: {targets.shape}")
    return sequences, targets # Return sequences and targets for immediate use if needed


if __name__ == "__main__":
    print("Generating and saving dummy data...")
    # This will generate 1000 raw records and save them to 'data/train.csv'
    # It will then create sequences and targets from this raw data using utils.create_sequences
    X_train, y_train = save_generated_data('data/train.csv', num_records=1000, seq_length=10)

    # You might want to save X_train and y_train directly as numpy files for faster loading
# Example for validation/test data
    print("\nGenerating and saving dummy validation data...")
    X_val, y_val = save_generated_data('data/val.csv', num_records=200, seq_length=10)
    np.save('data/X_val.npy', X_val) # Uncomment if you want to save as .npy
    np.save('data/y_val.npy', y_val) # Uncomment if you want to save as .npy
    print("X_val and y_val saved as .npy files.")

    print("\nGenerating and saving dummy test data...")
    X_test, y_test = save_generated_data('data/test.csv', num_records=200, seq_length=10)
    np.save('data/X_test.npy', X_test) # Uncomment if you want to save as .npy
    np.save('data/y_test.npy', y_test) # Uncomment if you want to save as .npy
    print("X_test and y_test saved as .npy files.")