import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class LSTMAttention(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 50
        self.batch_size = 500

    def _build_model(self) -> None:
        inputs = L.Input(shape=(self.input_timesteps, self.input_features))

        lstm100 = L.LSTM(units=100, return_sequences=True)(inputs)
        attention = L.Dense(1, activation='tanh')(lstm100)
        attention = L.Activation('softmax')(attention)

        concat = L.Concatenate()([lstm100, attention])
        predictions = L.Dense(self.output_timesteps, activation="linear")(concat)
        self.predictive_model = M.Model(inputs=inputs, outputs=predictions)

    def _build_model_obs(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.LSTM(units=100, input_shape=(self.input_timesteps, self.input_features), return_sequences=True))
        self.predictive_model.add(L.LSTM(units=100, return_sequences=True))
        self.predictive_model.add(L.Attention())
        self.predictive_model.add(L.Dropout(rate=0.2))
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="linear"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])