import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model


class LSTMModel:
    PREDITORS = [
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "changePercent",
        "vwap",
        "return",
        "return_squared",
        "weighted_average_sentiment",
        "h.1",
        "h.2",
        "h.3",
        "h.4",
        "h.5",
    ]

    def __init__(self, data, response_variable, split_ratio, window_size, n_steps):
        self.data = data.dropna(subset=[response_variable]).reset_index(drop=True)
        self.scaler = MinMaxScaler()
        self.split_index = int(len(self.data) * split_ratio)
        self.response_variable = response_variable
        self.window_size = window_size
        self.n_steps = n_steps

    def reshape_data(self, df):
        X_scaled = self.scaler.fit_transform(df[self.PREDITORS].to_numpy())
        # X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        y = df["realized_vol"].to_numpy()

        return X_scaled, y

    def create_sequences(self, X_scaled, y):
        X_seq, y_seq = [], []

        for i in range(
            len(X_scaled) - self.window_size - self.n_steps + 1
        ):  # Adjust the range to account for the forward-looking time step
            X_seq.append(X_scaled[i : i + self.window_size])

            y_seq.append(y[i + self.window_size + self.n_steps - 1])

        return np.array(X_seq), np.array(y_seq)

    def split_train_test_sequences(self, X_seq, y_seq):
        X_train, X_test = X_seq[: self.split_index], X_seq[self.split_index :]
        y_train, y_test = y_seq[: self.split_index], y_seq[self.split_index :]

        return X_train, X_test, y_train, y_test

    def create_and_fit_model(self, X_train, y_train):
        inputs = Input(shape=(self.window_size, len(self.PREDITORS)))
        lstm = LSTM(50, activation="relu")(inputs)
        outputs = Dense(1)(lstm)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train,
            y_train,
            epochs=50,
        )

        return model

    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
        print(y_pred.shape)
        return y_pred

    def eval_performance(self, y_test, y_pred):
        return mean_squared_error(y_test, y_pred)

    def insert_predictions(self, df, y_pred):
        df["predicted_vol"] = np.nan
        df.iloc[self.split_index + self.window_size + self.n_steps - 1 :, -1] = y_pred
        return df

    def lstm_modelling(self):
        X_scaled, y = self.reshape_data(self.data)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        X_train, X_test, y_train, y_test = self.split_train_test_sequences(X_seq, y_seq)
        model = self.create_and_fit_model(X_train, y_train)
        y_pred = self.predict(model, X_test)
        mse = self.eval_performance(y_test, y_pred)
        print(f"MSE: {mse}")
        df = self.insert_predictions(self.data, y_pred)
        return df

    def plot_forecasted_volatility(self, df):
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x=df.index,
            y="realized_vol",
            label="Realized Volatility",
        )
        sns.lineplot(
            data=df,
            x=df.index,
            y="predicted_vol",
            label="Predicted Volatility",
        )
        plt.title("Realized vs Predicted Volatility")
        plt.savefig("./images/lstm_forecast.png")


# sample usage
if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/hanyuwu/Study/stock_volatility_prediction/data/processed/final_data_w_garch.csv"
    )
    model = LSTMModel(data, "realized_vol", 0.8, 5, 5)
    df = model.lstm_modelling()
    df.to_csv("./data/processed/final_data_w_garch_lstm.csv", index=False)
    model.plot_forecasted_volatility(df)
