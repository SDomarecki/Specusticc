import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.predictive_models.neural_networks.model_v2 import ModelV2


class LSTMAttention(ModelV2):
    def __init__(self):
        super().__init__()
        self.epochs = 50
        self.batch_size = 500

    def _build_model(self) -> None:
        inputs = L.Input(shape=(100, 2))

        lstm100 = L.LSTM(units=100, return_sequences=True)(inputs)
        attention = L.Dense(1, activation='tanh')(lstm100)
        attention = L.Activation('softmax')(attention)

        concat = L.Concatenate()([lstm100, attention])
        predictions = L.Dense(50, activation="linear")(concat)
        self.predictive_model = M.Model(inputs=inputs, outputs=predictions)

    def _build_model_obs(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.LSTM(units=100, input_shape=(100, 2), return_sequences=True))
        self.predictive_model.add(L.LSTM(units=100, return_sequences=True))
        self.predictive_model.add(L.Attention())
        self.predictive_model.add(L.Dropout(rate=0.2))
        self.predictive_model.add(L.Dense(units=50, activation="linear"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])