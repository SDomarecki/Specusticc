import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.predictive_models.neural_networks.neural_network import NeuralNetwork


class LSTM(NeuralNetwork):
    def __init__(self, config: dict):
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