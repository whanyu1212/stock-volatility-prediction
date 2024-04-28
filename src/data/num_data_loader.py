import pandas as pd
import requests


class NumericalDataLoader:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        api_key: str,
        window_size: int = 5,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.window_size = window_size
        self.url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}"
        self.data = self.load_and_process_data()

    def load_data(self) -> pd.DataFrame:
        """parse the data from the API and
        return it as a DataFrame

        Returns:
            pd.DataFrame: raw data from the API
        """
        response = requests.get(self.url).json()
        df = pd.DataFrame(response["historical"])
        return df

    def convert_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the date column to datetime format

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with date column
            in datetime format
        """
        df["date"] = pd.to_datetime(df["date"])
        return df

    def sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort the dataframe by date

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe sorted by date
        """
        return df.sort_values(by="date").reset_index(drop=True)

    def calculate_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the return of the stock

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with return column
        """
        df["return"] = df["close"].pct_change()
        df.dropna(inplace=True)
        return df

    def calculate_return_squared(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the squared return of the stock,
        for SVR modelling later on

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with return_squared column
        """
        df["return_squared"] = df["return"] ** 2
        return df

    def calculate_realized_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the realized volatility of the stock

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with realized_vol column
        """
        df["realized_vol"] = df["return"].rolling(window=self.window_size).std()
        return df

    def calculate_implied_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the implied volatility of the stock by
        using the rolling standard deviation of the return
        after reversing the return column

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with implied_vol column
        """
        df["implied_vol"] = (
            df["return"][::-1].rolling(window=self.window_size).std()[::-1]
        )
        return df

    def trim_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim the dataframe to only include the necessary columns

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with only the necessary columns
        """
        return df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjClose",
                "changePercent",
                "volume",
                "vwap",
                "return",
                "return_squared",
                "realized_vol",
                "implied_vol",
            ]
        ]

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by converting the date column,
        sorting the dataframe by date, calculating the return,
        return squared, and realized volatility, and trimming
        the dataframe to only include the necessary columns

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: preprocessed dataframe
        """
        df = self.convert_date(df)
        df = self.sort_by_date(df)
        df = self.calculate_return(df)
        df = self.calculate_return_squared(df)
        df = self.calculate_realized_vol(df)
        df = self.calculate_implied_vol(df)
        df = self.trim_dataframe(df)
        return df

    def load_and_process_data(self) -> pd.DataFrame:
        """Load and preprocess the data

        Returns:
            pd.DataFrame: preprocessed data
        """
        raw_data = self.load_data()
        preprocessed_data = self.preprocess_data(raw_data)
        return preprocessed_data


# sample usage
if __name__ == "__main__":
    ticker = "NVDA"
    start_date = "2018-01-01"
    end_date = "2024-04-27"
    api_key = "API_KEY"
    data_loader = NumericalDataLoader(ticker, start_date, end_date, api_key)
    data = data_loader.data
    data.to_csv("./data/raw/numerical_data.csv", index=False)
    print(data.head())
