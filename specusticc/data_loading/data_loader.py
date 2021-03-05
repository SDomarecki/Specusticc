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

        if self._config.datasource == 'mongodb':
            self._datasource_name = 'MongoDB'
            self._set_collection()
        elif self._config.datasource == 'csv':
            self._datasource_name = 'CSV file'

    def _set_collection(self):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        stocks_database = client['stocks']
        self._collection = stocks_database['prices']

    def get_data(self) -> LoadedData:
        return self._dataset

    def load_data(self):
        self._load_data_dict('input')
        self._load_data_dict('output')
        self._load_data_dict('context')

    def _load_data_dict(self, dict_name: str):
        data_dict = {}
        if dict_name == 'input':
            tickers = self._config.input_tickers
        elif dict_name == 'output':
            tickers = self._config.output_tickers
        else:
            tickers = self._config.context_tickers

        for ticker in tickers:
            print(f'Loading {ticker} from {self._datasource_name} as {dict_name} data...', end='')
            loaded = self._load_one_dataframe(ticker)
            data_dict[ticker] = loaded
            print('Loaded')

        if dict_name == 'input':
            self._dataset.input = data_dict
        elif dict_name == 'output':
            self._dataset.output = data_dict
        else:
            self._dataset.context = data_dict

    # Loads whole financial history of one ticker
    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.lower()
        if ticker == 'test':
            raw_history = Generator.generate_test_data()
            history: pd.DataFrame = pd.DataFrame(raw_history)
        elif self._config.datasource == 'mongodb':
            raw_history = self._collection.find_one({"ticker": ticker})['history']
            history: pd.DataFrame = pd.DataFrame(raw_history)
        else:
            filepath = f'./database/{ticker}_d.csv'
            history = pd.read_csv(filepath)
            history.columns = ['date', 'open', 'high', 'low', 'close', 'vol']

        # transforms string date to datetime type
        history['date'] = pd.to_datetime(history['date'], format='%Y-%m-%d')
        return history
