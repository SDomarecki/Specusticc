from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


# see: Attention Is All You Need, url:https://arxiv.org/abs/1706.03762
class ModelTransformer(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        raise NotImplementedError

    def _build_model(self) -> None:
        pass

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
