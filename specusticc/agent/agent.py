from specusticc.data_processing.data_processor import DataProcessor
from specusticc.predictive_models.predictive_model_builder import PredictiveModelBuilder
from .load_config import load_config
from .reporting.reporter import Reporter


class Agent:
    def __init__(self, config_path: str, model_name: str) -> None:
        self.config = load_config(config_path)
        self.config['model']['name'] = model_name
        self.model = None
        self.data = None
        self.predictions = None

        self._boost_config_with_model()
        self._create_model_from_config()
        self._load_data()
        self._train_model()
        self._test_model()
        self._print_report()

    def _boost_config_with_model(self):
        simple_dim_list = ['basic', 'mlp']
        if self.config['model']['name'] in simple_dim_list:
            self.config['model']['input_dim'] = 2
        else:
            self.config['model']['input_dim'] = 3

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
