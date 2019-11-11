import tensorflow.keras.models as M
import tensorflow.keras.layers as L

from specusticc.neural_networks.model import Model

class NeuralNetworkBuilder:
    """A class for an building keras sequential model"""
    @staticmethod
    def build_network(configs: dict) -> Model:
        model = M.Sequential()
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            kernel_size = layer['kernel_size'] if 'kernel_size' in layer else 2
            pool_size = layer['pool_size'] if 'pool_size' in layer else None

            if layer['type'] == 'input':
                model.add(L.Input(shape=(input_timesteps, input_dim)))
            elif layer['type'] == 'dense':
                model.add(L.Dense(neurons, activation=activation))
            elif layer['type'] == 'lstm':
                model.add(L.LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            elif layer['type'] == 'dropout':
                model.add(L.Dropout(dropout_rate))
            elif layer['type'] == 'conv1d':
                model.add(L.Conv1D(neurons, kernel_size))
            elif layer['type'] == 'averagePooling1d':
                model.add(L.AveragePooling1D(pool_size))
            else:
                raise NotImplementedError

        model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        return Model(model, configs)
