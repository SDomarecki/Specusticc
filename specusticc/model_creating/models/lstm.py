import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model.agent_config import AgentConfig


class LSTM:
    def __init__(self, config: AgentConfig):
        self.epochs = 50

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features
        self.output_timesteps = config.output_timesteps

        self.possible_parameters = {}
        self._fetch_possible_parameters()

    def _fetch_possible_parameters(self):
        batch_size = [10, 50]
        optimizer = ['Adam']
        neurons = [20, 100]
        dropout_rate = [0.2, 0.8]

        self.possible_parameters = dict(
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                neurons=neurons)

    def build_model(self,
                    optimizer='adam',
                    dropout_rate=0.0,
                    neurons=100):
        model = M.Sequential()

        model.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        model.add(L.LSTM(units=neurons, return_sequences=True))
        model.add(L.LSTM(units=neurons, return_sequences=True))
        model.add(L.Dropout(rate=dropout_rate))
        model.add(L.LSTM(units=neurons, return_sequences=False))
        model.add(L.Dropout(rate=dropout_rate))
        model.add(L.Dense(units=self.output_timesteps, activation="linear"))

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model