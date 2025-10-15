# Medication Dispenser using RNN, GRU and Stacked LSTM

## Project Aim

This project aims to develop an intelligent medication dispenser system that predicts the risk of medication non-adherence using advanced machine learning models. By analyzing sequences of medication dispensing data, the system can forecast potential adherence issues, enabling timely interventions to improve patient outcomes and reduce healthcare costs.

## How It Helps People

Medication non-adherence is a major public health challenge, contributing to poor health outcomes, increased hospitalizations, and higher mortality rates. This project addresses this issue by:

- **Predicting Adherence Risk**: Using historical dispensing patterns to identify patients at risk of missing doses.
- **Enabling Proactive Care**: Allowing healthcare providers to intervene early with reminders, counseling, or adjusted treatment plans.
- **Improving Patient Outcomes**: Reducing the likelihood of complications from untreated conditions or improper medication use.
- **Supporting Personalized Medicine**: Tailoring interventions based on individual patient behavior patterns.
- **Reducing Healthcare Costs**: Preventing costly emergency interventions through predictive analytics.

The system can be integrated into smart pill dispensers, mobile apps, or healthcare platforms to provide real-time risk assessments and personalized support.

## Technology Stack

This project is built using modern machine learning and web development technologies:

- **Programming Language**: Python 3.7+
- **Machine Learning Framework**: TensorFlow with Keras for building and training RNN models
- **Model Architectures**:
  - Recurrent Neural Networks (RNN)
  - Gated Recurrent Units (GRU)
  - Stacked Long Short-Term Memory (LSTM)
- **Data Processing**: NumPy, Pandas, scikit-learn for data manipulation and preprocessing
- **Web Framework**: FastAPI for creating RESTful APIs
- **ASGI Server**: Uvicorn for running the API server
- **Data Validation**: Pydantic for request/response modeling
- **Serialization**: JSON for API communication

## Features

- Synthetic data generation for training and testing
- Multiple RNN model implementations with comparison capabilities
- Automated model training with early stopping and class weighting
- Comprehensive model evaluation metrics
- RESTful API for real-time predictions
- Modular, object-oriented codebase for easy extension

## Getting Started

For detailed instructions on setting up the environment, generating data, training models, testing, and deploying the API, please refer to `README_RUN.md`.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
