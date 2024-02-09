import streamlit as st
import pandas as pd
import numpy as np

# Streamlit page configuration
st.title('Power Meter Data Forecasting')

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not uploaded:
#     st.text('Please upload a CSV file to proceed.')

def preprocess_data(file):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Assuming the relevant power meter data is in a column named 'power'
    power_data = df['power'].values
    
    # Convert to numpy sequences of 99 for RNN
    sequences = []
    for i in range(len(power_data) - 99):
        sequences.append(power_data[i:i+99])
        
    return np.array(sequences)

if uploaded_file is not None:
    data = preprocess_data(uploaded_file)
    st.write('Data preprocessed and ready for prediction.')
