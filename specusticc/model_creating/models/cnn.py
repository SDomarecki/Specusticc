import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model.agent_config import AgentConfig


class CNN:
    def __init__(self, config: AgentConfig):
        self.epochs = 100

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features + config.context_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [20, 50, 100]
        epochs = [10, 25, 50, 100]
        optimizer = ['Adam']
        neurons = [20, 50, 100, 150, 200]
        activation = ['relu', 'softmax', 'linear', 'tanh']
        dropout_rate = [0.2, 0.4, 0.6, 0.8]
        kernel_size = [2, 4, 6, 8, 10]

        self.possible_parameters = dict(
            batch_size=batch_size,
            epochs=epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            neurons=neurons,
            kernel_size=kernel_size,
            activation=activation)

    def build_model(self,
                    optimizer='adam',
                    dropout_rate=0.1,
                    neurons=200,
                    kernel_size=3,
                    pool_size=2,
                    activation='relu',
                    conv_stacks=5):
        model = M.Sequential()

        model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        for i in range(conv_stacks):
            model.add(L.Conv1D(filters=neurons, kernel_size=kernel_size, activation=activation))
            model.add(L.Conv1D(filters=neurons, kernel_size=kernel_size, activation=activation))
            model.add(L.AveragePooling1D(pool_size=pool_size))
            model.add(L.Dropout(rate=dropout_rate))
        model.add(L.Flatten())
        model.add(L.Dense(units=neurons, activation=activation))
        model.add(L.Dense(units=self.output_timesteps, activation="linear"))

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model
