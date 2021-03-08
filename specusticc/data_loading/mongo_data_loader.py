import pymongo
import pandas as pd
from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.data_loading.data_loader import DataLoader


class MongoDataLoader(DataLoader):
    __mongo_path = "mongodb://localhost:27017/"

    def __init__(self, config: LoaderConfig):
        super().__init__(config)
        self.__datasource_name = "MongoDB"
        self.__collection = None

        self._init_collection()

    def _init_collection(self):
        client = pymongo.MongoClient(self.__mongo_path)
        stocks_database = client["stocks"]
        self.__collection = stocks_database["prices"]

    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        raw_history = self.__collection.find_one({"ticker": ticker})["history"]
        history: pd.DataFrame = pd.DataFrame(raw_history)

        # transforms string date to datetime type
        history["date"] = pd.to_datetime(history["date"], format="%Y-%m-%d")
        return history
