import pandas as pd
import pymongo

from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.data_loading.generator import Generator
from specusticc.data_loading.loaded_data import LoadedData


class DataLoader:
    def __init__(self, config: LoaderConfig):
        self._config = config
        self._collection = None
        self._dataset = LoadedData()

        self._set_collection()

    def _set_collection(self):
        client = pymongo.MongoClient(self._config.database_url)
        stocks_database = client['stocks']
        self._collection = stocks_database['prices']

    def get_data(self) -> LoadedData:
        return self._dataset

    def load_data(self):
        self._load_input_data()
        self._load_output_data()
        # Context will be empty in most of cases
        if self._config.context_tickers:
            self._load_context_data()

    def _load_input_data(self):
        input_data = {}
        for ticker in self._config.input_tickers:
            print('[Loader] Loading %s from MongoDB as input data' % ticker)
            loaded = self._load_one_dataframe(ticker)
            input_data[ticker] = loaded
            print('[Loader] Loaded')
        self._dataset.input = input_data

    def _load_output_data(self):
        output_data = {}
        for ticker in self._config.output_tickers:
            print('[Loader] Loading %s from MongoDB as output data' % ticker)
            loaded = self._load_one_dataframe(ticker)
            output_data[ticker] = loaded
            print('[Loader] Loaded')
        self._dataset.output = output_data

    def _load_context_data(self):
        context_data = {}
        for ticker in self._config.context_tickers:
            print('[Loader] Loading %s from MongoDB as context data' % ticker)
            loaded = self._load_one_dataframe(ticker)
            context_data[ticker] = loaded
            print('[Loader] Loaded')
        self._dataset.context = context_data

    # Loads whole financial history of one ticker
    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.lower()
        if ticker == 'test':
            raw_history = Generator.generate_test_data()
        else:
            raw_history = self._collection.find_one({"ticker": ticker})['history']

        history: pd.DataFrame = pd.DataFrame(raw_history)
        # transforms string date to datetime type
        history['date'] = pd.to_datetime(history['date'], format='%Y-%m-%d')
        return history
