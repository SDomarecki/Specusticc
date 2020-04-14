import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class LSTM(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 50
        self.batch_size = 100

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.LSTM(units=60, input_shape=(self.input_timesteps, self.input_features), return_sequences=True))
        self.predictive_model.add(L.LSTM(units=20, return_sequences=True))
        self.predictive_model.add(L.Dropout(rate=0.4))
        self.predictive_model.add(L.BatchNormalization())
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="softmax"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])