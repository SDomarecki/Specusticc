from specusticc.configs_init.configer import Configer
from specusticc.data_loading.data_loader import DataLoader
from specusticc.data_postprocessing.data_postprocessor import DataPostprocessor
from specusticc.data_processing.data_processor import DataProcessor
from specusticc.model_testing.tester import Tester
from specusticc.model_training.trainer import Trainer
from specusticc.model_creating.predictive_model_builder import PredictiveModelBuilder
from specusticc.reporting.reporter import Reporter


class Agent:
    def __init__(self, config_path: str, model_name: str):
        configer = Configer(config_path, model_name)
        self.configs = configer.get_class_configs()

        # Basic pipeline, probably to change when AutoML will be implemented
        self._load_data()  #1
        self._preprocess_data() #2
        self._create_predictive_model() #3
        self._train_model() #4
        self._test_model() #5
        self._postprocess_data() #6
        self._print_report() #7

    def _load_data(self):
        dl = DataLoader(self.configs['loader'])
        self.loaded_data = dl.get_data()

    def _preprocess_data(self):
        dp = DataProcessor(self.loaded_data, self.configs['preprocessor'])
        self.processed_data = dp.get_data()

    def _create_predictive_model(self):
        builder = PredictiveModelBuilder(self.configs['model_creator'])
        self.model = builder.build()

    def _train_model(self):
        trainer = Trainer(self.model, self.processed_data, self.configs['training'])
        trainer.train()
        self.model = trainer.get_model()

    def _test_model(self):
        tester = Tester(self.model, self.processed_data, self.configs['testing'])
        tester.test()
        self.test_results = tester.get_test_results()

    def _postprocess_data(self):
        dp = DataPostprocessor(self.processed_data, self.test_results, self.configs['postprocessor'])
        self.postprocessed_data = dp.get_data()

    def _print_report(self):
        r = Reporter(self.configs, self.postprocessed_data, self.model, self.configs['reporter'])
        r.print_report()
