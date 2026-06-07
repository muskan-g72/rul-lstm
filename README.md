## Remaining Useful Life (RUL) Prediction using LSTM

## Overview
This project predicts the Remaining Useful Life (RUL) of machinery using an LSTM-based deep learning model trained on time-series sensor data. It is designed for predictive maintenance to estimate when a machine is likely to fail.

##  Problem Statement
In industrial systems, unexpected machine failures lead to downtime and high costs. This project addresses this by predicting failure time using sequential sensor readings.

## Solution Approach
We use a Long Short-Term Memory (LSTM) neural network to capture temporal dependencies in sensor data and predict RUL values.

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib

## Dataset
- NASA CMAPSS dataset (or your dataset name)
- Multi-sensor time-series engine data

##  Model Architecture
- LSTM layers for sequence learning
- Dense layers for regression output
- Loss: Mean Squared Error (MSE)

## Workflow
1. Data preprocessing (cleaning + normalization)
2. Sequence generation for time-series modeling
3. LSTM model training
4. Evaluation using RMSE / MAE
5. Visualization of predictions

## Results
- RMSE: 13.15 
- MAE: 9.44
- R^2 Score: 0.892
- NASA Score: 323.7
- Improved prediction accuracy compared to baseline models

```md
## Project structure
rul-lstm/
│── notebooks/
│── src/
│── models/
│── data/
│── README.md

## How to Run
```bash
pip install -r requirements.txt
python src/train.py

---

## 📁 Project Structure
