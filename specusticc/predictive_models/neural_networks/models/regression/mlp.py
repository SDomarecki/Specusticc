import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.predictive_models.neural_networks.model_v2 import ModelV2


class MLP(ModelV2):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.Dense(units=500, activation="sigmoid", input_dim=200))
        self.predictive_model.add(L.Dense(units=200))
        self.predictive_model.add(L.Dropout(rate=0.5))
        self.predictive_model.add(L.Dense(50, activation="linear"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])