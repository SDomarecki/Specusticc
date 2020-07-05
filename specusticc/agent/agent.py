import logging

from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.data_postprocessing.data_postprocessor import DataPostprocessor
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.model_creating.trained_network_builder import TrainedNetworkBuilder
from specusticc.model_testing.tester import Tester
from specusticc.reporting.reporter import Reporter


class Agent:
    def __init__(self, model_name: str, fold_number: int, data: DataHolder, config: AgentConfig):
        self.config: AgentConfig = config

        self._name = f'{model_name}_{fold_number}'
        logging.info(f'Agent {self._name} start')

        self.model_name: str = model_name
        self._fold_number: int = fold_number
        self._data: DataHolder = data

    def run(self):
        self._create_and_train_model()
        self._test_model()

        self._postprocess_data()
        self._save_report()

    def _create_and_train_model(self):
        builder = TrainedNetworkBuilder(self._data, self.model_name, self.config)
        self._model = builder.build()

    def _test_model(self):
        tester = Tester(self._model, self.model_name, self._data)
        tester.test()
        self._test_results = tester.get_test_results()

    def _postprocess_data(self):
        postprocessor = DataPostprocessor(self._data, self._test_results)
        self._postprocessed_data = postprocessor.get_data()

    def _save_report(self):
        save_path = f'{self.config.market_save_path}/{self._name}'
        r: Reporter = Reporter(self._postprocessed_data, save_path)
        r.save_results()
