from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.decision_tree.decision_tree_builder import DecisionTreeBuilder
from specusticc.model_creating.neural_networks.neural_network_builder import NeuralNetworkBuilder


class PredictiveModelBuilder:
    def __init__(self, config: ModelCreatorConfig):
        self.config = config
        self.builder = None

    def build(self):
        self._create_model_from_config()
        return self.builder.build()

    def _create_model_from_config(self):
        model_type = self.config.model_type
        if model_type == 'neural_network':
            self._create_neural_network_builder()
        elif model_type == 'decision_tree':
            self._create_decision_tree_builder()
        else:
            raise NotImplementedError

    def _create_neural_network_builder(self):
        self.builder = NeuralNetworkBuilder(self.config)

    def _create_decision_tree_builder(self):
        self.builder = DecisionTreeBuilder(self.config)
