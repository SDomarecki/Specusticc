import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class MLP(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 200
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.Dense(units=500, activation="sigmoid", input_dim=self.input_timesteps*self.input_features))
        self.predictive_model.add(L.Dropout(rate=0.5))
        self.predictive_model.add(L.Dense(units=200))
        self.predictive_model.add(L.Dropout(rate=0.5))
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="softmax"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])