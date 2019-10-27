from keras import Sequential
from keras.layers import Dense, Dropout, LSTM

from specusticc.neural_networks.model import Model


class NeuralNetworkBuilder:
    """A class for an building keras sequential model"""
    @staticmethod
    def build_network(configs: dict) -> Model:
        model = Sequential()
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                model.add(Dropout(dropout_rate))

        model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        return Model(model)