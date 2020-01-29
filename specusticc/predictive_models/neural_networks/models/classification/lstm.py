import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.predictive_models.neural_networks.model_v2 import ModelV2


class LSTM(ModelV2):
    def __init__(self):
        super().__init__()
        self.epochs = 50
        self.batch_size = 100

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.LSTM(units=60, input_shape=(60, 5), return_sequences=True))
        self.predictive_model.add(L.LSTM(units=20, return_sequences=True))
        self.predictive_model.add(L.Dropout(rate=0.4))
        self.predictive_model.add(L.BatchNormalization())
        self.predictive_model.add(L.Dense(units=5, activation="softmax"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])