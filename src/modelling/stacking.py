import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from sklearn.svm import SVR
from src.modelling.lstm import (
    LSTMModel,
)  # change to from lstm import LSTMModel if running from this file
from src.modelling.svr import (
    SVRModel,
)  # change to from svr import SVRModel if running from this file


class StackingModel:
    def __init__(self, data, split_ratio, window_size, n_steps, response_variable):
        self.data = data.dropna(subset=[response_variable]).reset_index(drop=True)
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.n_steps = n_steps
        self.response_variable = response_variable
        self.lstm_model = LSTMModel(
            data, response_variable, split_ratio, window_size, n_steps
        )
        self.svr_model = SVRModel(data, split_ratio, n_steps, response_variable)

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataframe into training and testing sets.

        Parameters:
        df (pd.DataFrame): Dataframe to split

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes
        """
        split_index = int(len(df) * self.split_ratio)
        df_train = df[:split_index].reset_index(drop=True)
        df_test = df[split_index:].reset_index(drop=True)
        return df_train, df_test

    def fit_base_lstm(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Fit the base LSTM model and make predictions.

        Parameters:
        df_train (pd.DataFrame): Training dataframe

        Returns:
        np.ndarray: Predictions
        """
        X_train, y_train = self.lstm_model.reshape_data(df_train)
        X_seq, y_seq = self.lstm_model.create_sequences(X_train, y_train)
        model = self.lstm_model.create_and_fit_model(X_seq, y_seq)
        return self.lstm_model.predict(model, X_seq)

    def fit_base_svr(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Fit the base SVR model and make predictions.

        Parameters:
        df_train (pd.DataFrame): Training dataframe

        Returns:
        np.ndarray: Predictions
        """
        df_train_shifted = self.svr_model.created_shifted_response_variable(df_train)
        X_train, y_train = self.svr_model.generate_features_and_response(
            df_train_shifted
        )
        best_params = self.svr_model.optimize_hyperparameters(X_train, y_train)
        model = self.svr_model.fit_model(X_train, y_train, best_params)
        return self.svr_model.predict(model, X_train)[: -self.n_steps + 1]

    def combine_base_models(
        self,
        lstm_predictions: np.ndarray,
        svr_predictions: np.ndarray,
        training_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine the predictions of the base models.

        Parameters:
        lstm_predictions (np.ndarray): Predictions of the LSTM model
        svr_predictions (np.ndarray): Predictions of the SVR model
        training_data (pd.DataFrame): Training dataframe

        Returns:
        pd.DataFrame: Dataframe with combined predictions
        """
        prediction_dataframe = pd.DataFrame(
            {
                "lstm": lstm_predictions.flatten(),
                "svr": svr_predictions.flatten(),
            }
        )
        combined_dataframe = pd.concat(
            [
                training_data[[self.response_variable]].shift(-5),
                prediction_dataframe,
            ],
            axis=1,
            ignore_index=False,
        )
        return combined_dataframe

    def create_feature_response_for_stacking(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and response for the stacking model.

        Parameters:
        df (pd.DataFrame): Dataframe with combined predictions

        Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and response
        """
        X = df[["lstm", "svr"]]
        y = df[self.response_variable]
        return X, y

    def fit_stacking_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> LGBMRegressor:
        """
        Fit the stacking model.

        Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training response

        Returns:
        LGBMRegressor: Trained stacking model
        """
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        return model

    def predict(self, model: LGBMRegressor, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the stacking model.

        Parameters:
        model (LGBMRegressor): Trained stacking model
        X_test (pd.DataFrame): Test features

        Returns:
        np.ndarray: Predictions
        """
        return model.predict(X_test)

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Evaluate the performance of the stacking model.

        Parameters:
        y_true (pd.Series): True values
        y_pred (np.ndarray): Predictions

        Returns:
        float: Mean squared error
        """
        return mean_squared_error(y_true, y_pred)

    def train_stacking_model(self, training_data: pd.DataFrame) -> LGBMRegressor:
        """
        Train the stacking model.

        Parameters:
        training_data (pd.DataFrame): Training dataframe

        Returns:
        LGBMRegressor: Trained stacking model
        """
        lstm_predictions = self.fit_base_lstm(training_data)
        svr_predictions = self.fit_base_svr(training_data)
        combined_training_data = (
            self.combine_base_models(lstm_predictions, svr_predictions, training_data)
            .dropna()
            .reset_index(drop=True)
        )
        features, targets = self.create_feature_response_for_stacking(
            combined_training_data
        )
        trained_model = self.fit_stacking_model(features, targets)
        training_predictions = self.predict(trained_model, features)
        training_mse = self.evaluate(targets, training_predictions)
        print(f"Training MSE: {training_mse}")
        return trained_model

    def plot_volatility_prediction(self, y_true, y_pred, mse):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plt.suptitle(f"{self.response_variable} prediction with stacking")
        plt.title(f"Volatility Prediction with MSE: {mse}")
        plt.savefig(f"./images/stacking_performance_{self.response_variable}.png")

    def evaluate_stacking_model(
        self, trained_model: LGBMRegressor, testing_data: pd.DataFrame
    ) -> None:
        """
        Evaluate the performance of the stacking model on the testing data.

        Parameters:
        trained_model (LGBMRegressor): Trained stacking model
        testing_data (pd.DataFrame): Testing dataframe

        Returns:
        None
        """
        lstm_predictions = self.fit_base_lstm(testing_data)
        svr_predictions = self.fit_base_svr(testing_data)
        combined_testing_data = (
            self.combine_base_models(lstm_predictions, svr_predictions, testing_data)
            .dropna()
            .reset_index(drop=True)
        )
        features, targets = self.create_feature_response_for_stacking(
            combined_testing_data
        )
        testing_predictions = self.predict(trained_model, features)
        testing_mse = self.evaluate(targets, testing_predictions)
        print(f"Test MSE: {testing_mse}")
        self.plot_volatility_prediction(targets, testing_predictions, testing_mse)


if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/hanyuwu/Study/stock_volatility_prediction/data/processed/final_data_w_garch.csv"
    )
    stacker = StackingModel(data, 0.8, 5, 5, "implied_vol")
    df_train, df_test = stacker.split_train_test(stacker.data)
    model = stacker.train_stacking_model(df_train)
    stacker.evaluate_stacking_model(model, df_test)
