import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model.agent_config import AgentConfig


class BasicNet:
    def __init__(self, config: AgentConfig):
        self.epochs = 50

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [50]
        epochs = [25]
        optimizer = ['Nadam']
        neurons = [100, 150, 200]
        activation = ['relu']

        self.possible_parameters = dict(
                batch_size=batch_size,
                epochs=epochs,
                optimizer=optimizer,
                neurons=neurons,
                activation=activation)

    def build_model(self,
                    optimizer='nadam',
                    neurons=150,
                    activation='relu'):
        print(f'Optimizer={optimizer}, neurons={neurons}, activation={activation}')
        model = M.Sequential()

        model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        model.add(L.Flatten())
        model.add(L.Dense(units=neurons, activation=activation))
        model.add(L.Dense(self.output_timesteps, activation='linear'))

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model