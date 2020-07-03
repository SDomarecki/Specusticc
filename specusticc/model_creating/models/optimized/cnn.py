import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig


class CNN:
    def __init__(self, config: ModelCreatorConfig):
        self.epochs = 50
        self.batch_size = 500

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [10, 50]
        optimizer = ['Adam']
        neurons = [20, 100]
        activation = ['relu']
        dropout_rate = [0.2, 0.8]

        self.possible_parameters = dict(
            batch_size=batch_size,
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

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        return model
