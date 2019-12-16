from specusticc.data_processing.data_holder import DataHolder
from specusticc.data_processing.data_loader import DataLoader
from specusticc.predictive_models.decision_tree.tree_data_processor import TreeDataProcessor
from specusticc.predictive_models.neural_networks.neural_network_data_processor import NeuralNetworkDataProcessor


class DataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.train = None
        self.test = None
        self.data_holder = DataHolder()

    def create_data_holder(self) -> DataHolder:
        self._load_data()
        self._reshape_data_to_model_input_output()
        return self.data_holder

    def _load_data(self) -> None:
        loader = DataLoader(self.config)
        self.loaded_train_input, self.loaded_train_output, self.loaded_test_input, self.loaded_test_output = loader.load_data()
        self.data_holder.reporter_test_input = self.loaded_test_input.copy()

    def _reshape_data_to_model_input_output(self) -> None:
        type = self.config['model']['type']
        if type == 'decision_tree':
            data2io = TreeDataProcessor(self.config)
        elif type == 'neural_network':
            data2io = NeuralNetworkDataProcessor(self.config)
        else:
            raise NotImplementedError

        train_input, train_output = data2io.transform_for_model(self.loaded_train_input, self.loaded_train_output)
        test_input, test_output = data2io.transform_for_model(self.loaded_test_input, self.loaded_test_output)

        self.data_holder.train_input = train_input
        self.data_holder.train_output = train_output
        self.data_holder.test_input = test_input
        self.data_holder.test_output = test_output
