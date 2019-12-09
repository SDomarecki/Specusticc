from .load_config import load_config
from .reporting.reporter import Reporter
from specusticc.data_processing.data_processor import DataProcessor

from specusticc.predictive_models.predictive_model_builder import PredictiveModelBuilder


class Agent:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.model = None
        self.predictions = None
        self.input_train = None
        self.input_test = None
        self.output_train = None
        self.output_test = None

        self._create_model_from_config()
        self._load_data()
        self._train_model()
        self._test_model()
        self._print_report()

    def _create_model_from_config(self):
        builder = PredictiveModelBuilder(self.config)
        self.model = builder.build()

    def _load_data(self):
        dp = DataProcessor(self.config)
        self.input_train, self.output_train = dp.get_train_data()
        self.input_test, self.output_test = dp.get_test_data()
        self.original_pandas_input_test = dp.test

    def _train_model(self):
        self.model.train(self.input_train, self.output_train)

    def _test_model(self):
        if self.config['model']['target'] == 'regression':
            self.predictions = self.model.predict_regression(self.input_test)
        else:
            self.predictions = self.model.predict_classification(self.input_test)

    def _print_report(self):
        test = (self.original_pandas_input_test, self.output_test)
        r = Reporter(self.config, test, self.predictions, self.model)
        r.print_report()
