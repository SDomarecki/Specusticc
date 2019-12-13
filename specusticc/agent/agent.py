from .load_config import load_config
from .reporting.reporter import Reporter
from specusticc.data_processing.data_processor import DataProcessor

from specusticc.predictive_models.predictive_model_builder import PredictiveModelBuilder


class Agent:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.model = None
        self.data = None
        self.predictions = None

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
        self.data = dp.create_data_holder()

    def _train_model(self):
        self.model.train(self.data)

    def _test_model(self):
        self.predictions = self.model.predict(self.data.test_input)

    def _print_report(self):
        test = (self.data.reporter_test_input, self.data.test_output)
        r = Reporter(self.config, test, self.predictions, self.model)
        r.print_report()
