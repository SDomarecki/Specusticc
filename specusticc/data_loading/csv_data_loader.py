from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.data_loading.data_loader import DataLoader
import pandas as pd


class CSVDataLoader(DataLoader):
    def __init__(self, config: LoaderConfig):
        super().__init__(config)
        self.__database_path = config.database_path
        self.__datasource_name = "CSV file"

    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        filepath = f"{self.__database_path}/{ticker}_d.csv"
        history = pd.read_csv(filepath)
        history.columns = ["date", "open", "high", "low", "close", "vol"]
        history["date"] = pd.to_datetime(history["date"], format="%Y-%m-%d")
        return history
