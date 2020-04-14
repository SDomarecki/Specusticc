import pandas as pd
import pymongo

from specusticc.configs_init.loader_config import LoaderConfig
from specusticc.data_loading.generator import Generator


class DataLoader:
    def __init__(self, config: LoaderConfig) -> None:
        self.config = config

        self.collection = None
        self.input = {}
        self.output = {}

    def get_data(self) -> dict:
        self._set_collection()
        self._load_data()
        return {'input': self.input, 'output': self.output}

    def _set_collection(self):
        client = pymongo.MongoClient(self.config.database_url)
        stocks_database = client['stocks']
        self.collection = stocks_database['prices']

    def _load_data(self):
        self._load_input_data()
        self._load_output_data()

    def _load_input_data(self):
        for ticker in self.config.input_tickers:
            print('[Loader] Loading %s from MongoDB as input data' % ticker)
            loaded = self._load_one_dataframe(ticker)
            self.input[ticker] = loaded
            print('[Loader] Loaded')

    def _load_output_data(self):
        for ticker in self.config.output_tickers:
            print('[Loader] Loading %s from MongoDB as output data' % ticker)
            loaded = self._load_one_dataframe(ticker)
            self.output[ticker] = loaded
            print('[Loader] Loaded')

    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        if ticker == 'test':
            raw_history = Generator.generate_test_data()
        else:
            raw_history = self.collection.find_one({"ticker": ticker})['history']

        df_full_history = pd.DataFrame(raw_history)
        df_full_history['date'] = pd.to_datetime(df_full_history['date'], format='%Y-%m-%d')
        return df_full_history
