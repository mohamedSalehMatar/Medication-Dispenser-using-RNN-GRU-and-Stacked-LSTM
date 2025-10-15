import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from utils import create_sequences

# Define constants
MEDICINES = ['MedA', 'MedB', 'MedC', 'MedD', 'MedE']
label_encoder = LabelEncoder()
label_encoder.fit(MEDICINES)

def generate_daily_sequence(doses_per_day=3):
    sequence = []
    med = np.random.choice(MEDICINES)
    med_encoded = label_encoder.transform([med])[0]
    dose = np.random.randint(1, 5)

    interval = 24 * 3600 // doses_per_day
    base_time = 8 * 3600  # 08:00 AM

    for i in range(doses_per_day):
        scheduled_time = base_time + i * interval
        taken = np.random.rand() < 0.8  # 80% chance taken

        if taken:
            delay = np.random.randint(-600, 3600)  # ±10min to +1hr
            confirmation_time = scheduled_time + delay
            status = 1
        else:
            confirmation_time = 0
            status = 0

        sequence.append([
            med_encoded,
            dose,
            scheduled_time,
            confirmation_time,
            status
        ])
    return sequence

def generate_dataset(num_days=30000, doses_per_day=3):
    all_data = []
    for _ in range(num_days):
        all_data.extend(generate_daily_sequence(doses_per_day))
    df = pd.DataFrame(all_data, columns=[
        'medicine', 'dose', 'scheduled_time', 'confirmation_time', 'status'
    ])
    return df

def save_dataset_with_sequences(df, split_name, base_path='../dataset2', seq_length=3):
    os.makedirs(base_path, exist_ok=True)
    csv_path = os.path.join(base_path, f"{split_name}.csv")
    X_path = os.path.join(base_path, f"{split_name}_X.npy")
    y_path = os.path.join(base_path, f"{split_name}_y.npy")

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path} with {len(df)} rows.")

    # Create sequences
    X, y = create_sequences(df, seq_length=seq_length, input_dim=5)
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"Saved sequences to {X_path} and {y_path} — shapes: {X.shape}, {y.shape}")

if __name__ == "__main__":
    splits = [('train', 23332), ('val', 3334), ('test', 3334)]
    for split_name, days in splits:
        df = generate_dataset(num_days=days, doses_per_day=3)
        save_dataset_with_sequences(df, split_name)
