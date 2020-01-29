from specusticc.predictive_models.decision_tree.decision_tree_builder import DecisionTreeBuilder
from specusticc.predictive_models.neural_networks.neural_network_builder_v2 import NeuralNetworkBuilderV2


class PredictiveModelBuilder:
    def __init__(self, config: dict, model_name: str):
        self.model_name = model_name
        self.config = config
        self.builder = None

        self._create_model_from_config()

    def build(self):
        return self.builder.build()

    def _create_model_from_config(self):
        model_type = self.config['model']['type']
        if model_type == 'neural_network':
            self._create_neural_network_builder()
        elif model_type == 'decision_tree':
            self._create_decision_tree_builder()
        else:
            raise NotImplementedError

    def _create_neural_network_builder(self):
        target = self.config['model']['target']
        self.builder = NeuralNetworkBuilderV2(target, self.model_name)

    def _create_decision_tree_builder(self):
        self.builder = DecisionTreeBuilder(self.config)
