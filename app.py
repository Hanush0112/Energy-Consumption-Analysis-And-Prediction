import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

@st.cache_resource
def load_lstm_model():
    return load_model("energy_consumption_lstm_model.h5")

@st.cache_data
def load_data():
    return pd.read_csv("AEP_hourly.csv")

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

def predict(model, test_data, time_step=60):
    X_test = []
    for i in range(time_step, len(test_data)):
        X_test.append(test_data[i-time_step:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_data = model.predict(X_test)
    return predicted_data

def main():
    st.title("Energy Consumption Analysis and Prediction")

    dataset = load_data()

    dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])

    option = st.sidebar.selectbox("Choose Section", ["Analysis", "Prediction"])

    if option == "Analysis":
        st.header("Energy Consumption Analysis")
        
        year = st.selectbox("Select Year for Analysis", sorted(dataset['Datetime'].dt.year.unique()))

        data_year = dataset[dataset['Datetime'].dt.year == year]

        if len(data_year) == 0:
            st.warning(f"No data available for the year {year}")
        else:
            st.subheader(f"Energy Consumption for the Year {year}")

            plt.figure(figsize=(15, 6))
            plt.plot(data_year['Datetime'], data_year['AEP_MW'], color='green', label="Energy Consumption")
            plt.title(f'Energy Consumption for {year}')
            plt.xlabel('Datetime')
            plt.ylabel('Energy in MW')
            plt.legend()
            st.pyplot(plt)

            st.write("Summary Statistics for the Year:")
            st.write(data_year.describe())

    elif option == "Prediction":
        st.header("Energy Consumption Prediction")

        train_data, test_data, scaler = preprocess_data(dataset)

        model = load_lstm_model()

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
        st.pyplot(plt)

if __name__ == '__main__':
    main()
