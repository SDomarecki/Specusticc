import pandas as pd
from abc import ABC, abstractmethod
from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.data_loading.loaded_data import LoadedData


class DataLoader(ABC):
    def __init__(self, config: LoaderConfig):
        self.__datasource_name = ""
        self.__config = config
        self.__dataset = LoadedData()

    def get_data(self) -> LoadedData:
        return self.__dataset

    def load_data(self):
        self.__dataset.input = self._load_tickers(self.__config.input_tickers)
        self.__dataset.output = self._load_tickers(self.__config.output_tickers)
        self.__dataset.context = self._load_tickers(self.__config.context_tickers)

    def _load_tickers(self, tickers: str) -> dict:
        data_dict = {}

        for ticker in tickers:
            print(
                f"Loading {ticker} from {self.__datasource_name}...",
                end="",
            )
            ticker = ticker.lower()
            loaded = self._load_one_dataframe(ticker)
            data_dict[ticker] = loaded
            print("Loaded")

        return data_dict

    @abstractmethod
    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        """
        Loads whole financial history of one ticker
        :param ticker
        :return: Ticker price history
        """
        raise NotImplementedError
