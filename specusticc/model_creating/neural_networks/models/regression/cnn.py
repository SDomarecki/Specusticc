import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class CNN(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 50
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        self.predictive_model.add(L.Conv1D(filters=64, kernel_size=3, activation="sigmoid"))
        self.predictive_model.add(L.Conv1D(filters=64, kernel_size=3, activation="relu"))
        self.predictive_model.add(L.Dropout(rate=0.4))
        self.predictive_model.add(L.AveragePooling1D(pool_size=2))
        self.predictive_model.add(L.Flatten())
        self.predictive_model.add(L.Dense(units=100))
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="linear"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])