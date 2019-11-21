import json
from datetime import datetime

from .data_processing.data_processor import DataProcessor
from .reporting.reporter import Reporter
from .neural_networks.neural_network_builder import NeuralNetworkBuilder
from .decision_tree.decision_tree import DecisionTree


def load_config(config_path) -> dict:
    file = open(config_path)
    config = json.load(file)

    date = config['import']['date']
    date_format = '%Y-%m-%d'

    date['from'] = datetime.strptime(date['from'], date_format)
    date['to'] = datetime.strptime(date['to'], date_format)

    config['import']['date'] = date
    return config


class Agent:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.data_proc = None
        self.model = None
        self.predictions = None
        self.report = None
        self.input_train = None
        self.input_test = None
        self.output_train = None
        self.output_test = None

        self.load_data()
        self.create_model_from_config()
        self.prepare_data_batches()
        self.train_model()
        self.test_model()
        self.print_report()

    def load_data(self):
        self.data_proc = DataProcessor(self.config)

    def create_model_from_config(self):
        model_type = self.config['model']['type']
        if model_type == 'neural_network':
            self.create_neural_network_model()
        elif model_type == 'decision_tree':
            self.create_decision_tree_model()
        else:
            raise NotImplementedError

    def create_neural_network_model(self):
        self.model = NeuralNetworkBuilder.build_network(self.config)

    def create_decision_tree_model(self):
        self.model = DecisionTree(self.config)

    def prepare_data_batches(self):
        self.input_train, self.output_train = self.data_proc.get_train_data()
        self.input_test, self.output_test = self.data_proc.get_test_data()
        self.original_pandas_input_test = self.data_proc.test

    def train_model(self):
        self.model.train(self.input_train, self.output_train)

    def test_model(self):
        if self.config['model']['target'] == 'regression':
            self.predictions = self.model.predict_sequences_multiple(self.input_test)
        else:
            self.predictions = self.model.predict_classification(self.input_test)

    def print_report(self):
        test = (self.original_pandas_input_test, self.output_test)
        r = Reporter(self.config, test, self.predictions, self.model)
        r.print_report()
