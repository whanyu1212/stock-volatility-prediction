import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
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

    def reshape_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape the data to be used in the LSTM model

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray]: reshaped X and y arrays
        """
        X_scaled = self.scaler.fit_transform(df[self.PREDITORS].to_numpy())
        # X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        y = df["realized_vol"].to_numpy()

        return X_scaled, y

    def create_sequences(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for the LSTM model
        with the window size and forward-looking time step

        Args:
            X_scaled (np.ndarray): X array
            y (np.ndarray): y array

        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X_seq, y_seq = [], []

        for i in range(
            len(X_scaled) - self.window_size - self.n_steps + 1
        ):  # Adjust the range to account for the forward-looking time step
            X_seq.append(X_scaled[i : i + self.window_size])

            y_seq.append(y[i + self.window_size + self.n_steps - 1])

        return np.array(X_seq), np.array(y_seq)

    def split_train_test_sequences(
        self, X_seq: np.ndarray, y_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the sequences into training and testing sets

        Args:
            X_seq (np.ndarray): X sequences
            y_seq (np.ndarray): y sequences

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test = X_seq[: self.split_index], X_seq[self.split_index :]
        y_train, y_test = y_seq[: self.split_index], y_seq[self.split_index :]

        return X_train, X_test, y_train, y_test

    def create_and_fit_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tensorflow.keras.Model:
        """Create and fit the LSTM model
        Args:
            X_train (np.ndarray): training X array
            y_train (np.ndarray): training y array

        Returns:
            tensorflow.keras.Model: fitted LSTM model
        """
        inputs = Input(shape=(self.window_size, len(self.PREDITORS)))
        lstm = LSTM(100, activation="tanh", return_sequences=True)(
            inputs
        )  # Increase LSTM units and change activation function
        lstm = LSTM(50, activation="tanh")(lstm)  # Add another LSTM layer
        outputs = Dense(1)(lstm)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="RMSprop", loss="mse")  # Change optimizer

        model.fit(
            X_train,
            y_train,
            epochs=100,
        )

        return model

    def predict(self, model: tensorflow.keras.Model, X_test: np.ndarray) -> np.ndarray:
        """Make prediction on the test set

        Args:
            model (tensorflow.keras.Model): fitted LSTM model
            X_test (np.ndarray): test X array

        Returns:
            np.ndarray: predicted y array
        """
        y_pred = model.predict(X_test)
        return y_pred

    def eval_performance(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate the performance of the model using MSE

        Args:
            y_test (np.ndarray): actual y array
            y_pred (np.ndarray): predicted y array

        Returns:
            float: mse score
        """
        return mean_squared_error(y_test, y_pred)

    def insert_predictions(self, df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
        """Insert the predicted values back into the dataframe

        Args:
            df (pd.DataFrame): input dataframe
            y_pred (np.ndarray): predicted y array

        Returns:
            pd.DataFrame: dataframe with predicted values
        """
        df["predicted_vol"] = np.nan
        df.iloc[self.split_index + self.window_size + self.n_steps - 1 :, -1] = y_pred
        return df

    def lstm_modelling(self) -> Tuple[pd.DataFrame, float]:
        """Chains the methods to perform LSTM modelling

        Returns:
            Tuple[pd.DataFrame, float]: output DataFrame and MSE score
        """
        data = self.data.copy()
        X_scaled, y = self.reshape_data(data)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        X_train, X_test, y_train, y_test = self.split_train_test_sequences(X_seq, y_seq)
        model = self.create_and_fit_model(X_train, y_train)
        y_pred = self.predict(model, X_test)
        mse = self.eval_performance(y_test, y_pred)
        print(f"MSE: {mse}")
        df = self.insert_predictions(data, y_pred)
        return df, mse

    def plot_forecasted_volatility(self, df, mse):
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x=df.index,
            y=self.response_variable,
            label=self.response_variable,
        )
        sns.lineplot(
            data=df,
            x=df.index,
            y="predicted_vol",
            label="Predicted Volatility",
        )
        plt.suptitle(f"Forecasted {self.response_variable} with {self.n_steps} steps")
        plt.xlabel("Index")
        plt.ylabel("Volatility")
        plt.legend()
        plt.title(f"MSE: {mse}")
        plt.savefig(
            f"./images/lstm_forecasted_{self.response_variable}_{self.n_steps}_steps.png"
        )
        plt.close()


# sample usage
if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/hanyuwu/Study/stock_volatility_prediction/data/processed/final_data_w_garch.csv"
    )
    for n_steps in [1, 3, 5]:
        model = LSTMModel(data, "realized_vol", 0.8, 5, n_steps)
        df, mse = model.lstm_modelling()
        model.plot_forecasted_volatility(df, mse)
