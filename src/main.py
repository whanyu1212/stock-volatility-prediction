import os
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime
from src.data.text_data_loader import TextDataLoader
from src.data.num_data_loader import NumericalDataLoader
from src.data.text_data_transformation import TextTransformer
from src.data.data_stitching import DataStitcher
from src.modelling.garch import GarchModel
from src.modelling.lstm import LSTMModel
from src.modelling.svr import SVRModel
from src.modelling.stacking import StackingModel
from src.utils.general_util import parse_yaml_file

load_dotenv()
API_KEY = os.getenv("FMP_API_KEY")
os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"


def load_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    logger.info("Loading numerical data...")
    try:
        num_data_loader = NumericalDataLoader(ticker, start_date, end_date, API_KEY)
        num_data = num_data_loader.data
    except Exception as e:
        logger.error(f"Error loading numerical data: {e}")
    else:
        logger.success("Numerical data loaded successfully")

    logger.info("Loading text data...")
    try:
        text_data_loader = TextDataLoader(ticker, API_KEY)
        logger.info("Loading news data...")
        news_data = text_data_loader.load_news_data()
        logger.info("Loading press data...")
        press_data = text_data_loader.load_press_data()
        logger.info("Loading social media data...")
        twitter_data = text_data_loader.load_twitter_data()
    except Exception as e:
        logger.error(f"Error loading text data: {e}")
    else:
        logger.success("Text data loaded successfully")

    return num_data, news_data, press_data, twitter_data


def transform_data(news_data: pd.DataFrame, press_data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Transforming text data...")
    try:
        text_transformer = TextTransformer(news_data, "text")
        processed_news_data = text_transformer.transform()
        text_transformer = TextTransformer(press_data, "text")
        processed_press_data = text_transformer.transform()
    except Exception as e:
        logger.error(f"Error transforming text data: {e}")
    else:
        logger.success("Text data transformed successfully")

    processed_news_data.to_csv(
        "./data/intermediate/processed_news_data.csv", index=False
    )
    processed_press_data.to_csv(
        "./data/intermediate/processed_press_data.csv", index=False
    )

    return processed_news_data, processed_press_data


def stictch_data(
    num_data,
    processed_news_data,
    processed_press_data,
    start_date="2018-08-21",
    end_date="2024-04-27",
) -> pd.DataFrame:
    stitcher = DataStitcher(
        num_data,
        processed_news_data,
        processed_press_data,
        start_date,
        end_date,
    )

    stitched_data = stitcher.stitch_data()
    stitched_data.to_csv("./data/processed/final_data.csv", index=False)
    return stitched_data


def garch_flow(stitched_data):
    garch = GarchModel(stitched_data)
    result = garch.garch_modelling()
    garch.plot_forecasted_volatility(result)
    print(result)
    result.to_csv("./data/processed/final_data_w_garch.csv", index=False)
    return result


def run_lstm_model(data, variable, window_size, n_steps):
    model = LSTMModel(data, variable, 0.8, window_size, n_steps)
    df, mse = model.lstm_modelling()
    model.plot_forecasted_volatility(df, mse)


def lstm_flow(final_data_w_garch, window_size, n_steps_list):
    for n_steps in n_steps_list:
        run_lstm_model(final_data_w_garch, "realized_vol", window_size, n_steps)
        run_lstm_model(final_data_w_garch, "implied_vol", window_size, n_steps)


def run_svr_model(data, variable, n_steps):
    model = SVRModel(data, 0.8, n_steps, variable)
    df, mse = model.svr_modelling()
    model.plot_forecasted_volatility(df, mse)


def svr_flow(final_data_w_garch, n_steps_list):
    for n_steps in n_steps_list:
        run_svr_model(final_data_w_garch, "realized_vol", n_steps)
        run_svr_model(final_data_w_garch, "implied_vol", n_steps)


def run_stacking_model(data, variable):
    stacking_model = StackingModel(data, 0.8, 5, 5, variable)
    training_data, testing_data = stacking_model.split_train_test(stacking_model.data)
    trained_model = stacking_model.train_stacking_model(training_data)
    stacking_model.evaluate_stacking_model(trained_model, testing_data)


def stacking_flow(data):
    run_stacking_model(data, "realized_vol")
    run_stacking_model(data, "implied_vol")


def main():
    logger.info("Starting the pipeline")
    start_time = datetime.now()
    cfg = parse_yaml_file("./config/catalog.yaml")
    ticker, start_date, end_date = cfg["ticker"], cfg["start_date"], cfg["end_date"]
    window_size, n_steps_list = cfg["window_size"], cfg["n_steps"]

    num_data, news_data, press_data, social_media_data = load_data(
        ticker, start_date, end_date
    )

    processed_news_data, processed_press_data = transform_data(news_data, press_data)

    stitched_data = stictch_data(num_data, processed_news_data, processed_press_data)

    final_data_w_garch = garch_flow(stitched_data)

    logger.info("Running LSTM model")
    lstm_flow(final_data_w_garch, window_size, n_steps_list)

    logger.info("Running SVR model")
    svr_flow(final_data_w_garch, n_steps_list)

    logger.info("Running stacking model")
    stacking_flow(final_data_w_garch)

    end_time = datetime.now()
    logger.success(
        f"Pipeline completed successfully, time taken: {end_time-start_time}"
    )


if __name__ == "__main__":
    main()
