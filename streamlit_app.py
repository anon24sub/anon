import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import load

class QuantileRegressionPyTorchMSE(nn.Module):
    def __init__(self, input_length):
        super(QuantileRegressionPyTorchMSE, self).__init__()
        self.input_length = input_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(40 * (self.input_length - 10 - 6 + 2), 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, X, deterministic=False):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = self.dropout(X) if not deterministic else X
        X = torch.flatten(X, 1)
        X = F.relu(self.fc1(X))
        X = self.dropout(X) if not deterministic else X
        output = self.fc2(X)
        return output


def mse_loss(pred, y):
    return F.mse_loss(pred, y)
input_length = 99

model = QuantileRegressionPyTorchMSE(input_length=input_length)
model.load_state_dict(torch.load("model_cpu.pth"))
# Streamlit page configuration
st.title('Power Meter Data Forecasting')

def preprocess_data(file):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Assuming the relevant power meter data is in a column named 'power'
    power_data = df['Power Meter Reading'].values
    timestamp_data = df['timestamp'].values
    # Convert to numpy sequences of 99 for RNN
    sequences = []
    timestamps = []
    for i in range(len(power_data) - 99):
        sequences.append(power_data[i:i+99])
        timestamps.append(timestamp_data[i+98]) 
    return np.array(sequences).reshape(-1,1, 99),  np.array(timestamps)
st.write("\n\n")
st.write('This app is a demonstration of the model developed on the REDD Dataset.')
st.write("\n\n")
st.write('Please upload a csv file containing the timestamps and power meter readings. The name of column 1 should be "timestamp" and the name of column 2 should be "Power Meter Reading".')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
scaler_x = load(open('scaler_x.pkl', 'rb'))
scaler_y = load(open('scaler_y.pkl', 'rb'))

if uploaded_file is not None:
    power_data, timestamp = preprocess_data(uploaded_file)
    # print(data.shape)
    final_data = torch.tensor(power_data).float()
    print(final_data.shape)
    predictions = model(final_data)
    predictions = scaler_y.inverse_transform(predictions.detach().numpy())
    # preditions = model(torch.tensor(data[0]).float())
    timestamp = pd.to_datetime(timestamp)
    st.write('Data preprocessed and ready for prediction.')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plotting the predictions
    ax.plot(timestamp, predictions, label='Predicted')

    # Formatting the date on the X-axis to show at an angle
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Set interval here as needed
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)  # Rotate dates for better visibility

    # Adding labels and title
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Predictions', fontsize=14)
    plt.title('Predicted Power Meter Readings Over Time', fontsize=20)

    st.pyplot(fig)

