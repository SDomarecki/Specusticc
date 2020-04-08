import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.predictive_models.neural_networks.neural_network import NeuralNetwork


class CNN(NeuralNetwork):
    def __init__(self, config: dict):
        super().__init__(config)
        self.epochs = 100
        self.batch_size = 500

    def _build_model(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        self.predictive_model.add(L.Conv1D(filters=64, kernel_size=3, activation="relu"))
        self.predictive_model.add(L.Conv1D(filters=64, kernel_size=3, activation="relu"))
        self.predictive_model.add(L.Dropout(rate=0.4))
        self.predictive_model.add(L.AveragePooling1D(pool_size=2))
        self.predictive_model.add(L.Flatten())
        self.predictive_model.add(L.Dense(units=100))
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="softmax"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])