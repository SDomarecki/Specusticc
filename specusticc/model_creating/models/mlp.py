import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_network import NeuralNetwork


class MLP(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 200
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.Dense(units=200, input_dim=self.input_timesteps*self.input_features))
        self.predictive_model.add(L.Dense(units=100))
        self.predictive_model.add(L.Dense(units=100))
        self.predictive_model.add(L.Dense(self.output_timesteps))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])