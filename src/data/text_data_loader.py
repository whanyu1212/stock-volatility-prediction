import pandas as pd
import requests
from tqdm import tqdm


class TextDataLoader:
    def __init__(
        self,
        ticker: str,
        api_key: str,
        pages: int = 300,
    ):
        self.ticker = ticker
        self.api_key = api_key
        self.pages = pages
        self.news_url = f"https://financialmodelingprep.com/api/v3/stock_news"
        self.press_url = f"https://financialmodelingprep.com/api/v3/press-releases"
        self.social_media_url = (
            f"https://financialmodelingprep.com/api/v4/historical/social-sentiment"
        )

    def load_news_data(self) -> pd.DataFrame:
        """Load news data from the API page
        by page and return a DataFrame

        Returns:
            pd.DataFrame: output DataFrame
            with text and publishedDate columns
        """
        data_list = []
        for page in tqdm(range(self.pages)):
            response = requests.get(
                f"{self.news_url}?tickers={self.ticker}&page={page}&apikey={self.api_key}"
            )
            for item in response.json():
                data_list.append(
                    {"text": item["text"], "publishedDate": item["publishedDate"]}
                )

        return pd.DataFrame(data_list)

    def load_press_data(self) -> pd.DataFrame:
        """Load press data from the API page
        by page and return a DataFrame

        Returns:
            pd.DataFrame: output DataFrame
            with text and publishedDate columns
        """
        data_list = []
        for page in tqdm(range(self.pages)):
            response = requests.get(
                f"{self.press_url}/{self.ticker}?page={page}&apikey={self.api_key}"
            )
            for item in response.json():
                data_list.append({"text": item["title"], "publishedDate": item["date"]})

        return pd.DataFrame(data_list)

    def load_twitter_data(self) -> pd.DataFrame:
        """Load social media data such as stocktwits
        and twitter data from the API page by page

        Returns:
            pd.DataFrame: output DataFrame
            with social media data and processed
            sentiment columns
        """
        data_list = []
        for page in tqdm(range(self.pages)):
            response = requests.get(
                f"{self.social_media_url}?symbol={self.ticker}&page={page}&apikey={self.api_key}"
            )
            for item in response.json():
                data_list.append(
                    {
                        "date": item["date"],
                        "symbol": item["symbol"],
                        "stocktwitsPosts": item["stocktwitsPosts"],
                        "twitterPosts": item["twitterPosts"],
                        "stocktwitsComments": item["stocktwitsComments"],
                        "twitterComments": item["twitterComments"],
                        "stocktwitsLikes": item["stocktwitsLikes"],
                        "twitterLikes": item["twitterLikes"],
                        "stocktwitsImpressions": item["stocktwitsImpressions"],
                        "twitterImpressions": item["twitterImpressions"],
                        "stocktwitsSentiment": item["stocktwitsSentiment"],
                        "twitterSentiment": item["twitterSentiment"],
                    }
                )

        return pd.DataFrame(data_list)


# sample usage
if __name__ == "__main__":
    loader = TextDataLoader(
        "NVDA",
        "apikey",
    )
    news_data = loader.load_news_data()
    press_data = loader.load_press_data()
    twitter_data = loader.load_twitter_data()
    news_data.to_csv("./data/raw/news_data.csv", index=False)
    press_data.to_csv("./data/raw/press_data.csv", index=False)
    twitter_data.to_csv("./data/raw/twitter_data.csv", index=False)
    print(news_data.head())
    print(press_data.head())
    print(twitter_data.head())
