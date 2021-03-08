from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.data_loading.data_loader import DataLoader

import pandas as pd

from specusticc.data_loading.generator import Generator


class GeneratedDataLoader(DataLoader):
    def __init__(self, config: LoaderConfig):
        super().__init__(config)
        self.__datasource_name = "Artificial generator"

    def _load_one_dataframe(self, ticker: str) -> pd.DataFrame:
        raw_history = Generator.generate_test_data()
        history = pd.DataFrame(raw_history)
        return history
