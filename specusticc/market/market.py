from specusticc.agent.agent import Agent
from specusticc.configs_init.model.configs_wrapper import ConfigsWrapper
from specusticc.data_loading.csv_data_loader import CSVDataLoader
from specusticc.data_loading.generated_data_loader import GeneratedDataLoader
from specusticc.data_loading.loaded_data import LoadedData
from specusticc.data_loading.mongo_data_loader import MongoDataLoader
from specusticc.data_preprocessing.data_preprocessor import DataPreprocessor
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData


class Market:
    def __init__(self, configs: ConfigsWrapper, models: [str]):
        self._models: [str] = models
        self._configs: ConfigsWrapper = configs
        self.loaded_data: LoadedData
        self.preprocessed_data: PreprocessedData

    def run(self):
        self._load_data()
        self._preprocess_data()
        self._run_agents()

    def _load_data(self):
        if self._configs.loader.datasource == "mongodb":
            dl = MongoDataLoader(self._configs.loader)
        elif self._configs.loader.datasource == "csv":
            dl = CSVDataLoader(self._configs.loader)
        else:
            dl = GeneratedDataLoader(self._configs.loader)
        dl.load_data()
        self.loaded_data = dl.get_data()

    def _preprocess_data(self):
        dp = DataPreprocessor(self._configs.preprocessor)
        dp.preprocess_data(self.loaded_data)
        self.preprocessed_data = dp.get_data()

    def _run_agents(self):
        n_folds: int = self._configs.market.n_folds
        for model in self._models:
            for i in range(n_folds):
                agent = Agent(
                    model_name=model,
                    fold_number=i + 1,
                    data=self.preprocessed_data,
                    config=self._configs.agent,
                )
                agent.run()
