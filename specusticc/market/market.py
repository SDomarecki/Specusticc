import logging

from specusticc.agent.agent import Agent
from specusticc.configs_init.configer import Configer
from specusticc.configs_init.model.configs_wrapper import ConfigsWrapper
from specusticc.data_loading.data_loader import DataLoader
from specusticc.data_loading.loaded_data import LoadedData
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.data_preprocessing.data_preprocessor import DataPreprocessor

import specusticc.utilities.directories as dirs


class Market:
    def __init__(self, config_path: str, models: [str]):
        logging.info('Market start')

        timestamp = dirs.get_timestamp()
        self.market_save_path = f'output/{timestamp}'
        dirs.create_save_dir(self.market_save_path)

        self._models: [str] = models
        configer = Configer(config_path, self.market_save_path)
        self._configs: ConfigsWrapper = configer.get_configs_wrapper()

        self.loaded_data: LoadedData
        self.processed_data: DataHolder

    def run(self):
        self._load_data()
        self._preprocess_data()
        self._run_agents()

    def _load_data(self):
        dl = DataLoader(self._configs.loader)
        dl.load_data()
        self.loaded_data = dl.get_data()

    def _preprocess_data(self):
        dp = DataPreprocessor(self.loaded_data, self._configs.preprocessor)
        dp.preprocess_data()
        self.processed_data = dp.get_data()

    def _run_agents(self):
        n_folds: int = self._configs.market.n_folds
        for model in self._models:
            for i in range(n_folds):
                agent: Agent = Agent(model_name=model,
                                     fold_number=i+1,
                                     data=self.processed_data,
                                     config=self._configs.agent)
                agent.run()