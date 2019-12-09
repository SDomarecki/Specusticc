from specusticc.data_processing.data_loader import DataLoader
from specusticc.predictive_models.decision_tree.tree_data_processor import TreeDataProcessor
from specusticc.predictive_models.neural_networks.neural_network_data_processor import NeuralNetworkDataProcessor


class DataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.train = None
        self.test = None

        self._load_data()
        self._reshape_data_to_model_input_output()

    def _load_data(self) -> None:
        loader = DataLoader(self.config)
        self.loaded_train_input, self.loaded_train_output, self.loaded_test_input, self.loaded_test_output = loader.load_data()
        self.test = self.loaded_test_input.copy()

    def _reshape_data_to_model_input_output(self):
        type = self.config['model']['type']
        if type == 'decision_tree':
            data2io = TreeDataProcessor(self.config)
        elif type == 'neural_network':
            data2io = NeuralNetworkDataProcessor(self.config)
        else:
            raise NotImplementedError

        self.input_train, self.output_train = data2io.transform_for_model(self.loaded_train_input, self.loaded_train_output)
        self.input_test, self.output_test = data2io.transform_for_model(self.loaded_test_input, self.loaded_test_output)

    def get_train_data(self) -> tuple:
        return self.input_train, self.output_train

    def get_test_data(self) -> tuple:
        return self.input_test, self.output_test
