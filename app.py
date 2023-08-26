import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Download Historical Stock Data
@st.cache_data
def download_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load Data
st.title('Stock Price Prediction with LSTM')
symbol = st.text_input('Enter Stock Symbol (e.g., UNVR.JK):')
start_date = st.date_input('Select Start Date:')
end_date = st.date_input('Select End Date:')

if st.button('Download Data'):
    data = download_data(symbol, start_date, end_date)
    st.write(data.head())

    # Preprocess Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(100, len(train_data)):
        x_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build and Train LSTM Model
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=5, epochs=50)

    # Create Test Data
    test_data = scaled_data[training_data_len - 100:, :]

    x_test = []
    y_test = data['Close'][training_data_len:].values

    for i in range(100, len(test_data)):
        x_test.append(test_data[i-100:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluation
    st.write("### RMSE :")
    rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
    rmse

    st.write("### MAPE :")
    errors = np.abs(predictions - y_test)
    mape = np.mean(errors / y_test) * 100
    mape

    # Plot the Results
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    st.write("### Stock Price Prediction vs Actual")
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price Rupiah (Rp)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
    st.pyplot(plt)

    valid[['Close', 'Predictions']]
 
