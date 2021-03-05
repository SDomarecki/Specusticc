from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.data_postprocessing.data_postprocessor import DataPostprocessor
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData
from specusticc.model_creating.network_builder import NetworkBuilder
from specusticc.model_creating.optimizer import Optimizer
from specusticc.model_testing.tester import Tester
from specusticc.model_training.trainer import Trainer
from specusticc.reporting.reporter import Reporter


class Agent:
    def __init__(
        self,
        model_name: str,
        fold_number: int,
        data: PreprocessedData,
        config: AgentConfig,
    ):
        self.config: AgentConfig = config

        self._name = f"{model_name}_{fold_number}"
        self.model_name: str = model_name
        self._fold_number: int = fold_number
        self._data: PreprocessedData = data

    def run(self):
        hyperparam_method = self.config.hyperparam_optimization_method
        if hyperparam_method in ["grid", "random", "bayes"]:
            self._optimize_model()
        else:
            self._create_model()
            self._train_model()
        self._test_model()

        self._postprocess_data()
        self._save_report()

    def _optimize_model(self):
        optimizer = Optimizer(self._data, self.model_name, self.config)
        self._model = optimizer.optimize()

    def _create_model(self):
        builder = NetworkBuilder(self.model_name, self.config)
        self._model = builder.build()

    def _train_model(self):
        trainer = Trainer(self._data, self.model_name, self.config)
        self._model = trainer.train(self._model)

    def _test_model(self):
        tester = Tester(self._model, self.model_name, self._data)
        tester.test()
        self._test_results = tester.get_test_results()

    def _postprocess_data(self):
        postprocessor = DataPostprocessor(self._data, self._test_results)
        self._postprocessed_data = postprocessor.get_data()

    def _save_report(self):
        save_path = self.config.save_path
        reporter = Reporter(self._postprocessed_data, save_path, self._name)
        reporter.save_results()
