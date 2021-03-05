from specusticc.agent.agent import Agent
from specusticc.configs_init.configer import Configer
from specusticc.configs_init.model.configs_wrapper import ConfigsWrapper
from specusticc.data_loading.data_loader import DataLoader
from specusticc.data_loading.loaded_data import LoadedData
from specusticc.data_preprocessing.data_preprocessor import DataPreprocessor
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData


class Market:
    def __init__(self, example_path: str, models: [str]):
        self._models: [str] = models
        configer = Configer(save_path=example_path)
        configer.create_all_configs()
        self._configs: ConfigsWrapper = configer.get_configs_wrapper()

        self.loaded_data: LoadedData
        self.preprocessed_data: PreprocessedData

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
