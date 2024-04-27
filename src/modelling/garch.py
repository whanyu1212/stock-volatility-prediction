import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from arch import arch_model


class GarchModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_best_params(self, df):
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

    def fit(self, df, best_param):
        optimal_model = arch_model(
            df["return"] * 10,
            mean="zero",
            vol="GARCH",
            p=best_param[0],
            o=0,
            q=best_param[1],
        ).fit(disp="off")

        return optimal_model

    def predict(self, optimal_model, df):
        forecast_values = optimal_model.forecast(start=0, horizon=5).variance
        return (np.sqrt(forecast_values)) / 10

    def concatenate_data(self, df, forecast_values):
        df = pd.concat([df, forecast_values], axis=1)
        return df

    def garch_modelling(self):
        df = self.data.copy()
        best_param = self.get_best_params(df)
        optimal_model = self.fit(df, best_param)
        forecast_values = self.predict(optimal_model, df)
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
