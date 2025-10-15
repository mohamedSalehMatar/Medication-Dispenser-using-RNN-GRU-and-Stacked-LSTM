# Medication Dispenser Model: Complete Guide

This documentation provides a comprehensive, step-by-step guide to running the Medication Dispenser project from data generation to model training, testing, deploying the API server, and requesting predictions. The project uses RNN-based models (Stacked LSTM, GRU, and RNN) to predict medication adherence risk based on dispensing sequences.

## Table of Contents

1. [Prerequisites and Setup](#prerequisites-and-setup)
2. [Step 1: Data Generation](#step-1-data-generation)
3. [Step 2: Training the Models](#step-2-training-the-models)
4. [Step 3: Testing the Models](#step-3-testing-the-models)
5. [Step 4: Running the API Server](#step-4-running-the-api-server)
6. [Step 5: Requesting Predictions](#step-5-requesting-predictions)
7. [Additional Notes](#additional-notes)

## Prerequisites and Setup

### System Requirements
- Python 3.7 or higher
- Sufficient disk space for datasets and models (at least 1GB recommended)

### Dependencies
Install the required Python packages using pip:

```bash
pip install tensorflow keras numpy scikit-learn fastapi uvicorn pydantic pandas
```

Alternatively, if a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

### Project Structure
Ensure your project directory is set up as follows:

```
med-dispense-model/
├── app/
│   ├── generate_data.py
│   ├── train.py
│   ├── test.py
│   ├── model_api_server.py
│   ├── model_stacked_lstm.py
│   ├── model_gru.py
│   ├── model_rnn.py
│   ├── utils.py
│   └── data/  # Will be created during data generation
├── trained_model/  # Will be created during training
└── README_RUN.md
```

## Step 1: Data Generation

The first step is to generate synthetic data simulating medication dispensing sequences. This data includes features like medicine type, dose, scheduled time, confirmation time, and status (taken or not).

### Running Data Generation

Navigate to the `app` directory and run the data generation script:

```bash
cd app
python generate_data.py
```

This script will:
- Generate datasets for training, validation, and testing splits.
- Create CSV files (`train.csv`, `val.csv`, `test.csv`) in the `../dataset2` directory (relative to `app`).
- Convert the data into sequences and save NumPy arrays (`train_X.npy`, `train_y.npy`, etc.) for efficient loading.

### Example Output
```
Saved CSV to ../dataset2/train.csv with 69996 rows.
Saved sequences to ../dataset2/train_X.npy and ../dataset2/train_y.npy — shapes: (69996, 3, 5), (69996, 1)
```

### Customization
- To change the number of days or doses per day, edit the `generate_dataset` call in `generate_data.py`.
- Default: 23,332 days for training, 3,334 for validation and testing, 3 doses per day.

## Step 2: Training the Models

Train one of the RNN-based models (Stacked LSTM, GRU, or RNN) on the generated data.

### Running Training

From the `app` directory:

```bash
python train.py
```

You will be prompted to select a model:

```
Select model to train:
1. Stacked LSTM
2. GRU
3. RNN
Enter choice (1, 2, 3):
```

Enter `1`, `2`, or `3` to choose the model.

The script will:
- Load or generate the training data.
- Build and compile the selected model.
- Train the model with early stopping and class weighting.
- Save the best model to `../trained_model/model_{model_type}.keras`.

### Example Output
```
Loading training, validation, and test data from pre-generated numpy files...
Building Stacked LSTM model...
Training complete. Best model saved to ../trained_model/model_stacked_lstm.keras
```

### Customization
- Modify parameters like epochs, batch size, or sequence length in the `Trainer` class instantiation in `train.py`.
- Default: 100 epochs, batch size 32, sequence length 3.

## Step 3: Testing the Models

Evaluate the trained model on the test dataset to assess performance.

### Running Testing

From the `app` directory:

```bash
python test.py
```

You will be prompted to select a model:

```
Select model to test:
1. Stacked LSTM
2. GRU
3. RNN
Enter choice (1 - 3):
```

Enter `1`, `2`, or `3` to choose the model.

The script will:
- Load the specified model from `../trained_model/`.
- Load the test data.
- Evaluate and print metrics: Accuracy, Precision, Recall, F1-Score.
- Display predictions for the first 20 samples.

### Example Output
```
Loading test data from pre-generated numpy files...
Sample 1: Predicted Status = 1 (Prob = 0.8523), True Status = 1
...
Accuracy:  0.923456
Precision: 0.876543
Recall:    0.912345
F1-Score:  0.894123
```

### Customization
- Adjust the number of test samples or model path in the `Tester` class in `test.py`.
- Default: Uses all test sequences.

## Step 4: Running the API Server

Deploy the trained models via a FastAPI server for real-time predictions.

### Running the Server

From the `app` directory:

```bash
python model_api_server.py
```

The server will start on `http://0.0.0.0:8000` and load the trained models (GRU, LSTM, RNN) if available.

### Example Output
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

If a model fails to load, it will print an error but continue running.

### API Endpoints
- `POST /predict_gru`: Predict using GRU model
- `POST /predict_lstm`: Predict using Stacked LSTM model
- `POST /predict_rnn`: Predict using RNN model

## Step 5: Requesting Predictions

Send requests to the running server with sequence data to get risk predictions.

### Example Request

Use a tool like `curl` or Postman to send a POST request with JSON data.

#### Using curl (from command line):

```bash
curl -X POST "http://localhost:8000/predict_lstm" \
     -H "Content-Type: application/json" \
     -d '{
       "sequence": [
         [0.0, 2.0, 28800.0, 29100.0, 1.0],
         [0.0, 2.0, 43200.0, 43500.0, 1.0],
         [0.0, 2.0, 57600.0, 0.0, 0.0]
       ]
     }'
```

#### Explanation of Input:
- `sequence`: A list of 3 lists (sequence length), each with 5 features:
  - Medicine (encoded, e.g., 0 for MedA)
  - Dose (e.g., 2)
  - Scheduled time (seconds since midnight, e.g., 28800 for 8:00 AM)
  - Confirmation time (0 if not taken)
  - Status (1 if taken, 0 otherwise)

#### Example Response:
```json
{
  "model": "Stacked LSTM",
  "risk_prediction": 0.123456
}
```

- `risk_prediction`: A float between 0 and 1 indicating the predicted risk of non-adherence (lower is better adherence).

### Error Handling
- If the model is not loaded: `{"detail": "LSTM model not loaded"}`
- Invalid input: `{"detail": "Prediction error: ..."}`

## Additional Notes

- **Data Format**: Sequences are of length 3 (timesteps), with 5 features per timestep.
- **Models**: All models output a probability for adherence risk.
- **Troubleshooting**: Ensure data is generated before training/testing. Check file paths if errors occur.
- **Extensibility**: Modify model architectures in `model_*.py` files or data generation in `generate_data.py`.
- **Performance**: Training may take time depending on hardware; use GPUs if available.

For issues or contributions, refer to the main README.md or contact the maintainers.
