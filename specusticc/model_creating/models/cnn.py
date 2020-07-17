import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model.agent_config import AgentConfig


class CNN:
    def __init__(self, config: AgentConfig):
        self.epochs = 50

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features + config.context_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [50]
        epochs = [25]
        optimizer = ['Adam']
        neurons = [20, 100]
        activation = ['relu']
        dropout_rate = [0.2, 0.8]

        self.possible_parameters = dict(
            batch_size=batch_size,
            epochs=epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            neurons=neurons,
            activation=activation)

    def build_model(self,
                    optimizer='adam',
                    dropout_rate=0.0,
                    neurons=20,
                    activation='relu'):
        model = M.Sequential()

        model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        model.add(L.Conv1D(filters=neurons, kernel_size=3, activation=activation))
        model.add(L.Conv1D(filters=neurons, kernel_size=3, activation=activation))
        model.add(L.Dropout(rate=dropout_rate))
        model.add(L.AveragePooling1D(pool_size=2))
        model.add(L.Flatten())
        model.add(L.Dense(units=neurons))
        model.add(L.Dense(units=self.output_timesteps, activation="linear"))

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model
