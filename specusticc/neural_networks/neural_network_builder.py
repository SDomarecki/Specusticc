import tensorflow.keras.models as M
import tensorflow.keras.layers as L

from specusticc.neural_networks.model import Model


class NeuralNetworkBuilder:
    """A class for an building keras sequential model"""

    @staticmethod
    def build_network(config: dict) -> Model:
        model = M.Sequential()
        for layer in config['model']['layers']:
            model.add(NeuralNetworkBuilder.build_layer(layer))

        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        print('[Model] Model Compiled')
        return Model(model, config)

    @staticmethod
    def build_layer(config_layer: dict):
        neurons = config_layer['neurons'] if 'neurons' in config_layer else None
        dropout_rate = config_layer['rate'] if 'rate' in config_layer else None
        activation = config_layer['activation'] if 'activation' in config_layer else None
        return_seq = config_layer['return_seq'] if 'return_seq' in config_layer else None
        input_timesteps = config_layer['input_timesteps'] if 'input_timesteps' in config_layer else None
        input_dim = config_layer['input_dim'] if 'input_dim' in config_layer else None
        kernel_size = config_layer['kernel_size'] if 'kernel_size' in config_layer else 2
        pool_size = config_layer['pool_size'] if 'pool_size' in config_layer else None

        layer_type = config_layer['type']
        if layer_type == 'input':
            return L.Input(shape=(input_timesteps, input_dim))
        elif layer_type == 'dense':
            return L.Dense(neurons, activation=activation)
        elif layer_type == 'lstm':
            return L.LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq)
        elif layer_type == 'gru':
            return L.GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq)
        elif layer_type == 'attention':
            return L.Attention()
        elif layer_type == 'dropout':
            return L.Dropout(dropout_rate)
        elif layer_type == 'conv1d':
            return L.Conv1D(neurons, kernel_size)
        elif layer_type == 'averagePooling1d':
            return L.AveragePooling1D(pool_size)
        elif layer_type == 'batchNormalization':
            return L.BatchNormalization()
        else:
            raise NotImplementedError
