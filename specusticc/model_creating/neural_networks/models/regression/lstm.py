import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class LSTM(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 20
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.LSTM(units=100, input_shape=(self.input_timesteps, self.input_features), return_sequences=True))
        self.predictive_model.add(L.LSTM(units=100, return_sequences=True))
        self.predictive_model.add(L.Dropout(rate=0.2))
        self.predictive_model.add(L.LSTM(units=100, return_sequences=False))
        self.predictive_model.add(L.Dropout(rate=0.2))
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="linear"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])