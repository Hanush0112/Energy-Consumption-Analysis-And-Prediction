import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

dataset = pd.read_csv("AEP_hourly.csv")

dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])

def preprocess_data(dataset, train_size=0.8):
    data = dataset["AEP_MW"].values.reshape(-1, 1)  

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_data_len = int(len(scaled_data) * train_size)
    train_data = scaled_data[:train_data_len]
    test_data = scaled_data[train_data_len:]

    return train_data, test_data, scaler

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, Y_train, epochs=20, batch_size=32):
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    return model

def predict(model, test_data, time_step=60):
    X_test = []
    for i in range(time_step, len(test_data)):
        X_test.append(test_data[i-time_step:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_data = model.predict(X_test)
    return predicted_data

train_data, test_data, scaler = preprocess_data(dataset)

X_train, Y_train = create_dataset(train_data)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = train_model(X_train, Y_train)

predicted_data = predict(model, test_data)

predicted_data = scaler.inverse_transform(predicted_data)
true_data = scaler.inverse_transform(test_data[60:]) 

plt.figure(figsize=(15, 6))
plt.plot(dataset['Datetime'][-len(true_data):], true_data, color='green', label="True Energy Consumption")
plt.plot(dataset['Datetime'][-len(predicted_data):], predicted_data, color='red', label="Predicted Energy Consumption")
plt.title('Energy Consumption Prediction')
plt.xlabel('Datetime')
plt.ylabel('Energy in MW')
plt.legend()
plt.show()


model.save("energy_consumption_lstm_model.h5")
