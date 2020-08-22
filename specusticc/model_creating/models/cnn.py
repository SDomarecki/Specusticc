import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from specusticc.configs_init.model.agent_config import AgentConfig


class CNN:
    def __init__(self, config: AgentConfig):
        self.epochs = 200

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features + config.context_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [20, 50, 100]
        epochs = [50, 100]
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
                    dropout_rate=0.0,
                    neurons=100,
                    filters=50,
                    kernel_size=3,
                    pool_size=2,
                    activation='relu',
                    conv_stacks=4):

        input_layer = L.Input(shape=(self.input_timesteps, self.input_features))

        cnn1 = input_layer
        for i in range(conv_stacks):
            cnn1 = L.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(cnn1)
            cnn1 = L.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(cnn1)
            cnn1 = L.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(cnn1)
            cnn1 = L.MaxPooling1D(pool_size=pool_size)(cnn1)
            cnn1 = L.BatchNormalization()(cnn1)
            cnn1 = L.Dropout(rate=dropout_rate)(cnn1)
        cnn1 = L.Flatten()(cnn1)

        merge = cnn1
        dense = L.Dense(units=neurons, activation=activation)(merge)
        dense = L.Dense(units=neurons, activation=activation)(dense)
        dense = L.Dense(units=neurons, activation=activation)(dense)
        output = L.Dense(units=self.output_timesteps, activation="linear")(dense)
        model = M.Model(inputs=input_layer, outputs=output)

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model
