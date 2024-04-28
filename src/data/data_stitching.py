import pandas as pd
import numpy as np


class DataStitcher:
    def __init__(
        self,
        num_data: pd.DataFrame,
        news_data: pd.DataFrame,
        press_data: pd.DataFrame,
        left_bound: str,
        right_bound: str,
    ):
        self.num_data = self.convert_date_to_datetime(num_data)
        self.news_data = self.convert_date_to_datetime(news_data)
        self.press_data = self.convert_date_to_datetime(press_data)
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.date_df = self.create_date_df()

    def convert_date_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the date column from string to datetime

        Args:
            df (pd.DataFrame): input DataFrame

        Raises:
            ValueError: if the DataFrame does not have a 'date' column

        Returns:
            pd.DataFrame: output DataFrame with the 'date' column converted to datetime
        """
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            raise ValueError("DataFrame does not have a 'date' column.")
        return df

    def filter_data_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the data by date range

        Args:
            df (pd.DataFrame): input DataFrame

        Returns:
            pd.DataFrame: output DataFrame filtered by date range
        """
        return df[(df["date"] >= self.left_bound) & (df["date"] <= self.right_bound)]

    def create_date_df(self) -> pd.DataFrame:
        """Create a DataFrame with a date range
        to fill the missing dates in the final DataFrame

        Returns:
            pd.DataFrame: output DataFrame with just a date column
        """
        date_df = pd.DataFrame(
            {"date": pd.date_range(self.left_bound, self.right_bound)}
        )
        return date_df

    def calculate_weighted_average_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the weighted average sentiment of news and press data,
        store it in a new column, and fill the missing values with 0

        Args:
            df (pd.DataFrame): input DataFrame

        Returns:
            pd.DataFrame: output DataFrame with the weighted average sentiment column
        """
        df["weighted_average_sentiment"] = np.where(
            df["news_sentiment"].isna(),
            df["press_sentiment"],
            np.where(
                df["press_sentiment"].isna(),
                df["news_sentiment"],
                0.6 * df["news_sentiment"] + 0.4 * df["press_sentiment"],
            ),
        )
        df["weighted_average_sentiment"].fillna(0, inplace=True)
        return df

    def combine_news_press_data(self) -> pd.DataFrame:
        """Combine the news and press data with the numerical data

        Returns:
            pd.DataFrame: output DataFrame with the combined data
        """
        news_data_filtered = self.filter_data_by_date(self.news_data)
        press_data_filtered = self.filter_data_by_date(self.press_data)
        sentiment_df = (
            self.date_df.merge(news_data_filtered, on="date", how="left")
            .merge(press_data_filtered, on="date", how="left")
            .rename(
                columns={
                    "sentiment_score_x": "news_sentiment",
                    "sentiment_score_y": "press_sentiment",
                }
            )
        )
        sentiment_df = self.calculate_weighted_average_sentiment(sentiment_df)
        return sentiment_df

    def stitch_data(self) -> pd.DataFrame:
        """Flow function that connects all the methods

        Returns:
            pd.DataFrame: final data prior to modelling
        """
        sentiment_data = self.combine_news_press_data()
        final_data = self.num_data.merge(sentiment_data, on="date", how="inner")
        return final_data


# sample usage

if __name__ == "__main__":
    num_data = pd.read_csv("./data/raw/numerical_data.csv")
    news_data = pd.read_csv("./data/intermediate/processed_news_data.csv")
    press_data = pd.read_csv("./data/intermediate/processed_press_data.csv")

    stitcher = DataStitcher(
        num_data,
        news_data,
        press_data,
        "2018-08-21",
        "2024-04-27",
    )

    stitched_data = stitcher.stitch_data()

    print(stitched_data.head())

    stitched_data.to_csv("./data/processed/final_data.csv", index=False)
