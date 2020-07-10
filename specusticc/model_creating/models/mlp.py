import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model.agent_config import AgentConfig


class MLP:
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
        optimizer = ['adam']
        neurons = [200]
        neurons2 = [20]
        activation = ['relu']
        activation2 = ['relu']
        dropout_rate = [0.2]

        self.possible_parameters = dict(
            batch_size=batch_size,
            epochs=epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            neurons=neurons,
            neurons2=neurons2,
            activation=activation,
            activation2=activation2)

    def build_model(self,
                    optimizer='adam',
                    dropout_rate=0.2,
                    neurons=200,
                    neurons2=20,
                    activation='relu',
                    activation2='relu'):
        print(f'Optimizer={optimizer}, dropout_rate={dropout_rate}, neurons={neurons}, neurons2={neurons2}, activation={activation}, activation2={activation2}')
        model = M.Sequential()

        model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        model.add(L.Flatten())
        model.add(L.Dense(units=neurons, activation=activation))
        model.add(L.Dense(units=neurons2, activation=activation2))
        model.add(L.Dropout(rate=dropout_rate))
        model.add(L.Dense(units=neurons2, activation=activation))
        model.add(L.Dense(self.output_timesteps, activation='linear'))

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model
