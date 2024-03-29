import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizers

from specusticc.configs_init.model.agent_config import AgentConfig


class LSTM:
    def __init__(self, config: AgentConfig):
        self.epochs = 50

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features + config.context_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [20, 50, 100]
        epochs = [10, 25, 50, 100]
        optimizer = ["adam"]
        neurons = [100, 200]
        neurons2 = [20, 50, 100]
        dropout_rate = [0.2, 0.5, 0.8]

        self.possible_parameters = dict(
            batch_size=batch_size,
            epochs=epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            neurons=neurons,
            neurons2=neurons2,
        )

    def build_model(self, optimizer="adam", dropout_rate=0.1, neurons=100, neurons2=50):
        print(
            f"Optimizer={optimizer}, dropout_rate={dropout_rate}, neurons={neurons}, neurons2={neurons2}"
        )
        model = Models.Sequential()

        model.add(Layers.Input(shape=(self.input_timesteps, self.input_features)))
        model.add(Layers.LSTM(units=self.input_features, return_sequences=True))
        model.add(Layers.LSTM(units=5, return_sequences=True))
        model.add(Layers.Dropout(rate=dropout_rate))
        model.add(Layers.LSTM(units=1, return_sequences=True))
        model.add(Layers.Dropout(rate=dropout_rate))
        model.add(Layers.Flatten())
        model.add(Layers.Dense(units=self.output_timesteps, activation="linear"))

        opt = Optimizers.Adam(learning_rate=0.1)
        mape = "mean_absolute_percentage_error"
        model.compile(loss=mape, optimizer=opt, metrics=[mape])

        return model
