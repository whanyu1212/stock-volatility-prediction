import optuna
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class SVRModel:
    PREDITORS = [
        "return_squared",
        "h.1",
        "h.2",
        "h.3",
        "h.4",
        "h.5",
    ]

    def __init__(self, data, split_ratio, n_steps, response_variable):
        self.data = data.dropna(subset=["realized_vol", "implied_vol"]).reset_index(
            drop=True
        )
        self.split_index = int(len(self.data) * split_ratio)
        self.n_steps = n_steps
        self.response_variable = response_variable

    def created_shifted_response_variable(self, df):
        df["shifted_response_variable"] = df[self.response_variable].shift(
            -self.n_steps
        )
        return df.dropna(subset=["shifted_response_variable"]).reset_index(drop=True)

    def generate_features_and_response(self, df):
        X = df[self.PREDITORS]
        y = df["shifted_response_variable"]
        return X, y

    def split_train_test(self, X, y):
        X_train, X_test = X[: self.split_index], X[self.split_index :]
        y_train, y_test = y[: self.split_index], y[self.split_index :]
        return X_train, X_test, y_train, y_test

    def objective(self, trial, X_train, y_train):
        gamma = trial.suggest_loguniform("gamma", 1e-5, 1e2)
        C = trial.suggest_loguniform("C", 1e-5, 1e2)
        epsilon = trial.suggest_loguniform("epsilon", 1e-5, 1e2)
        svr = SVR(kernel="rbf", gamma=gamma, C=C, epsilon=epsilon)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_train)
        return mean_squared_error(y_train, y_pred)

    def optimize_hyperparameters(self, X_train, y_train):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train), n_trials=100
        )
        return study.best_params

    def fit_model(self, X_train, y_train, best_params):
        svr = SVR(kernel="rbf", **best_params)
        svr.fit(X_train, y_train)
        return svr

    def predict(self, model, X_test):
        return model.predict(X_test)

    def eval_performance(self, y_test, y_pred):
        return mean_squared_error(y_test, y_pred)

    def insert_predictions(self, df, y_pred):
        df["predicted_vol"] = np.nan
        df.loc[self.split_index :, "predicted_vol"] = y_pred
        return df

    def svr_modelling(self):
        df = self.data.copy()
        df = self.created_shifted_response_variable(df)
        X, y = self.generate_features_and_response(df)
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        best_params = self.optimize_hyperparameters(X_train, y_train)
        model = self.fit_model(X_train, y_train, best_params)
        y_pred = self.predict(model, X_test)
        mse = self.eval_performance(y_test, y_pred)
        print(f"MSE: {mse}")
        df = self.insert_predictions(df, y_pred)
        return df, mse

    def plot_forecasted_volatility(self, df, mse):
        df_plot = df.copy()
        plt.style.use("ggplot")
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_plot,
            x=df_plot.index,
            y=self.response_variable,
            label=self.response_variable,
        )
        sns.lineplot(
            data=df_plot, x=df_plot.index, y="predicted_vol", label="Predicted"
        )
        plt.suptitle(f"Forecasted {self.response_variable} with {self.n_steps} steps")
        plt.xlabel("Index")
        plt.ylabel("Volatility")
        plt.legend()
        plt.title(f"MSE: {mse}")
        plt.savefig(
            f"./images/svr_forecasted_{self.response_variable}_{self.n_steps}_steps.png"
        )
        plt.close()


# sample usage
if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/hanyuwu/Study/stock_volatility_prediction/data/processed/final_data_w_garch.csv"
    )
    for n_steps in [1, 3, 5]:
        model = SVRModel(data, 0.8, n_steps, "realized_vol")
        df, mse = model.svr_modelling()
        model.plot_forecasted_volatility(df, mse)
