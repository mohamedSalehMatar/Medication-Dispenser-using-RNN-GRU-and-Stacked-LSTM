import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

class Tester:
    def __init__(self, model_path="../trained_model/model.keras", seq_len=3, num_samples=100):
        self.model_path = model_path
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def prepare_data(self):
        import os
        import numpy as np
        import pandas as pd
        dataset_dir = '../dataset2'
        X_test_path = os.path.join(dataset_dir, 'test_X.npy')
        y_test_path = os.path.join(dataset_dir, 'test_y.npy')
        test_csv = os.path.join(dataset_dir, 'test.csv')

        if os.path.exists(X_test_path) and os.path.exists(y_test_path):
            print("Loading test data from pre-generated numpy files...")
            self.X_test = np.load(X_test_path, allow_pickle=True)
            self.y_test = np.load(y_test_path, allow_pickle=True)
        else:
            print("Pre-generated numpy test files not found, generating dataset from CSV...")
            if not os.path.exists(test_csv):
                print("Test CSV file not found. Please run generate_data.py first to create datasets.")
                raise FileNotFoundError("Required test CSV dataset file is missing.")
            test_df = pd.read_csv(test_csv)

            from utils import create_sequences
            self.X_test, self.y_test = create_sequences(test_df, seq_length=self.seq_len, input_dim=5)
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_test_path, self.X_test)
            np.save(y_test_path, self.y_test)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        pred_labels = (predictions >= 0.5).astype(int)

        accuracy = accuracy_score(self.y_test, pred_labels)
        precision = precision_score(self.y_test, pred_labels)
        recall = recall_score(self.y_test, pred_labels)
        f1score = f1_score(self.y_test, pred_labels)

        for i, (pred_prob, pred_label, true) in enumerate(zip(predictions[:20], pred_labels[:20], self.y_test[:20])):
            print(f"Sample {i+1}: Predicted Status = {int(pred_label[0])} (Prob = {pred_prob[0]:.4f}), True Status = {int(true[0])}")

        print(f"\nAccuracy:  {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall:    {recall:.6f}")
        print(f"F1-Score:  {f1score:.6f}")



if __name__ == "__main__":
    print("Select model to test:")
    print("1. Stacked LSTM")
    print("2. GRU")
    print("3. RNN")
    choice = input("Enter choice (1 - 3): ").strip()
    if choice == '1':
        model_type = 'stacked_lstm'
    elif choice == '2':
        model_type = 'gru'
    elif choice == '3':
        model_type = 'rnn'
    else:
        print("Invalid choice, defaulting to Stacked LSTM")
        model_type = 'stacked_lstm'

    model_path = f"../trained_model/model_{model_type}.keras"
    tester = Tester(model_path=model_path, num_samples=10000)
    tester.load_model()
    tester.prepare_data()
    tester.evaluate()
