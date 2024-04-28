import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from arch import arch_model
from arch.univariate.base import ARCHModelResult


class GarchModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_best_params(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Get the best combination of p and q for the GARCH model
        based on the BIC score

        Args:
            df (pd.DataFrame): input DataFrame

        Returns:
            Tuple[int, int]: best combination of p and q
        """
        bic_garch = []

        for p in range(1, 10):
            for q in range(1, 10):
                garch = arch_model(
                    df["return"] * 10, mean="zero", vol="GARCH", p=p, o=0, q=q
                ).fit(disp="off")
                bic_garch.append(garch.bic)
                if garch.bic == np.min(bic_garch):
                    best_param = p, q
        print(f"Best Parameters: {best_param}")
        return best_param

    def fit(self, df: pd.DataFrame, best_param: Tuple[int, int]) -> ARCHModelResult:
        """Fit the garch model with the best parameters

        Args:
            df (pd.DataFrame): input dataframe
            best_param (Tuple[int, int]): combination of p and q

        Returns:
            ARCHModelResult: fitted GARCH model
        """
        optimal_model = arch_model(
            df["return"] * 10,
            mean="zero",
            vol="GARCH",
            p=best_param[0],
            o=0,
            q=best_param[1],
        ).fit(disp="off")

        return optimal_model

    def predict(self, optimal_model: ARCHModelResult) -> pd.DataFrame:
        """Forecast the volatility using the optimal model,
        horizon of 5 days

        Args:
            optimal_model (ARCHModelResult): fitted GARCH model

        Returns:
            pd.DataFrame: dataframe with forecasted volatility h1-h5
        """
        forecast_values = optimal_model.forecast(start=0, horizon=5).variance
        return (np.sqrt(forecast_values)) / 10

    def concatenate_data(
        self, df: pd.DataFrame, forecast_values: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenate the forecasted values with the original DataFrame

        Args:
            df (pd.DataFrame): input DataFrame
            forecast_values (pd.DataFrame): forecasted values

        Returns:
            pd.DataFrame: output DataFrame with forecasted values
        """
        df = pd.concat([df, forecast_values], axis=1)
        return df

    def garch_modelling(self) -> pd.DataFrame:
        """Connects the methods to perform GARCH modelling

        Returns:
            pd.DataFrame: output DataFrame with forecasted values
        """
        df = self.data.copy()
        best_param = self.get_best_params(df)
        optimal_model = self.fit(df, best_param)
        forecast_values = self.predict(optimal_model)
        df = self.concatenate_data(df, forecast_values)
        return df

    def plot_forecasted_volatility(self, df):
        # Slice the DataFrame from index 5 onwards
        df_plot = df[5:]

        # Set the ggplot style
        plt.style.use("ggplot")

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_plot,
            x=df_plot.index,
            y="realized_vol",
            label="Realized Volatility",
            color="pink",
        )
        sns.lineplot(
            data=df_plot,
            x=df_plot.index,
            y=df["h.5"].shift(5)[5:],
            label="Forecasted Volatility",
            color="red",
        )

        plt.title("Actual vs Forecasted Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.savefig("./images/garch_forecasted_volatility.png")


# sample usage
if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/hanyuwu/Study/stock_volatility_prediction/data/processed/final_data.csv"
    )
    garch = GarchModel(data)
    result = garch.garch_modelling()
    garch.plot_forecasted_volatility(result)
    print(result)
    result.to_csv("./data/processed/final_data_w_garch.csv", index=False)
