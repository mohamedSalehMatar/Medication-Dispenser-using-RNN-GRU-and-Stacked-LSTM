# Running Training and Testing for Models

This document explains how to run training and testing for the two implemented models: Stacked LSTM and GRU.

## Setup

Ensure you have the following dependencies installed:

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- scikit-learn

You can install them via pip:

```bash
pip install tensorflow keras numpy scikit-learn
```

## Training

The training script is `app/train.py`. It supports training either the stacked LSTM or GRU model via an interactive prompt.

### Run Training

```bash
python app/train.py
```

You will be prompted to select the model to train:

```
Select model to train:
1. Stacked LSTM
2. GRU
Enter choice (1 or 2):
```

Enter `1` for Stacked LSTM or `2` for GRU.

The script trains the selected model on generated dummy data (default 5000 samples, 20 epochs) and saves the best model to:

- `trained_model/model_stacked_lstm.keras` for Stacked LSTM
- `trained_model/model_gru.keras` for GRU

### Customize Training

To customize parameters like number of samples, epochs, batch size, or model type, edit the `Trainer` instantiation in `app/train.py`.

## Testing

The testing script is `app/test.py`. It loads a saved model and evaluates it on generated test data.

### Run Testing

```bash
python app/test.py
```

You will be prompted to select the model to test:

```
Select model to test:
1. Stacked LSTM
2. GRU
Enter choice (1 or 2):
```

Enter `1` or `2` to load the corresponding saved model and run evaluation.

The script evaluates on 100 test samples by default and prints:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Sample predicted vs true risk scores for the first 10 samples

### Customize Testing

To customize parameters like test sample size or model path, edit the `Tester` instantiation in `app/test.py`.

## Data Generation and Risk Score

- Dummy data is generated with realistic feature ranges and slight anomalies.
- Risk scores are calculated using a weighted sum of normalized features.
- Risk scores are min-max normalized to [0,1] to match the model's sigmoid output.

## Notes

- Models are saved separately per model type to avoid confusion.
- The codebase uses an object-oriented design for modularity.
- You can extend or modify the models, data generation, and training/testing scripts as needed.

---

For any questions or issues, please reach out.
