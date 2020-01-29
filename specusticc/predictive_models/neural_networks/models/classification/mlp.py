import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.predictive_models.neural_networks.model_v2 import ModelV2


class MLP(ModelV2):
    def __init__(self):
        super().__init__()
        self.epochs = 200
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.Dense(units=500, activation="sigmoid", input_dim=500))
        self.predictive_model.add(L.Dropout(rate=0.5))
        self.predictive_model.add(L.Dense(units=200))
        self.predictive_model.add(L.Dropout(rate=0.5))
        self.predictive_model.add(L.Dense(units=5, activation="softmax"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])