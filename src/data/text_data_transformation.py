import pandas as pd
import spacy
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tqdm.pandas()


class TextTransformer:
    def __init__(self, data: pd.DataFrame, column: str):
        """Initializes the class with the data and the column

        Args:
            data (pd.DataFrame): input dataframe
            column (str): the column that contains the text data

        Raises:
            ValueError: raise an error if the data is not a pandas DataFrame
            ValueError: raise an error if the data has no string columns
            ValueError: raise an error if the selected column is not a string
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if not any(data.dtypes == "object"):
            raise ValueError("data must contain at least one column of type string")

        if not isinstance(column, str):
            raise ValueError("The selected must be a string variable")

        self.data = data
        self.column = column
        self.data[self.column] = self.data[self.column].astype(str)
        self.nlp = spacy.load("en_core_web_sm")

    def lower_case(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all the alphabets in the string of text
        to lower case

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: dataframe with lower case text
            in the column that was selected
        """
        df[self.column] = df[self.column].str.lower()
        return df

    def remove_special_characters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove everything that is not alphanumeric or spaces, tabs, line
        breaks from the text.

        \w: This matches any word character (equal to [a-zA-Z0-9_])
        \s: This matches any whitespace character (spaces, tabs, line breaks)
        [^...]: The caret ^ inside the square brackets negates the set,
        meaning it matches any character not in the set

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with special characters removed
        """
        df[self.column] = df[self.column].str.replace(r"[^\w\s]", "", regex=True)
        return df

    def strip_extra_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """remove extra spaces from the text

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with extra spaces removed from the
            text in the selected column
        """
        df[self.column] = df[self.column].str.strip().str.replace(" +", " ", regex=True)
        return df

    def apply_spacy_pipeline(self, text: str) -> list:
        """Apply the default spacy processing
        pipeline to the text

        Args:
            text (str): the string value in each
            row of the selected column

        Returns:
            list: a list of lemmatized tokens
        """
        doc = self.nlp(text)
        lemmatized_text = [token.lemma_ for token in doc if not token.is_stop]
        return lemmatized_text

    def preprocessing_flow(self, df) -> pd.DataFrame:
        """The main function that applies all the
        preprocessing steps. We keep the processed
        text as a list and as a string because we
        may need to use them in different ways

        Returns:
            pd.DataFrame: dataframe with
            the processed text columns
        """

        df = self.lower_case(df)
        df = self.remove_special_characters(df)
        df = self.strip_extra_spaces(df)
        df[f"Processed_{self.column}_list"] = df[self.column].progress_apply(
            self.apply_spacy_pipeline
        )
        # its easier to use string for count vectorizer and tfidf
        df[f"Processed_{self.column}"] = df[f"Processed_{self.column}_list"].apply(
            " ".join
        )
        return df

    def get_compound_sentiment_score(self, text: list) -> float:
        """Using the NLTK VADER sentiment analyzer
        to get the compound score for each piece
        of text

        Args:
            text (list): text stored in list

        Returns:
            float: score in the range of -1 to 1
        """
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]

    def apply_sentiment_analysis(self, df) -> pd.DataFrame:
        """Apply the sentiment analysis to the processed text

        Returns:
            pd.DataFrame: dataframe with the sentiment scores
        """
        df["sentiment_score"] = df[f"Processed_{self.column}"].progress_apply(
            self.get_compound_sentiment_score
        )
        return df

    def convert_to_datetime(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = pd.to_datetime(df[column])
        return df

    def extract_date(
        self, df: pd.DataFrame, source_column: str, target_column: str
    ) -> pd.DataFrame:
        df[target_column] = df[source_column].dt.date
        return df

    def aggregate_by_date(
        self, df: pd.DataFrame, date_column: str, agg_column: str
    ) -> pd.DataFrame:
        df_agg = df.groupby(date_column)[agg_column].median().reset_index()
        return df_agg

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.convert_to_datetime(df, "publishedDate")
        df = self.extract_date(df, "publishedDate", "date")
        df_agg = self.aggregate_by_date(df, "date", "sentiment_score")
        return df_agg

    def transform(self) -> pd.DataFrame:
        df = self.data.copy()
        df = self.preprocessing_flow(df)
        df = self.apply_sentiment_analysis(df)
        df = self.post_process(df)
        return df


# sample_usage

if __name__ == "__main__":
    news_data = pd.read_csv("./data/raw/news_data.csv")
    news_data["text"] = news_data["text"].astype(str)
    processed_news_data = TextTransformer(news_data, "text").transform()
    print(processed_news_data.head())
