import tensorflow.keras.models as M
import tensorflow.keras.layers as L

from specusticc.predictive_models.neural_networks.model import Model


class NeuralNetworkBuilder:
    """A class for an building keras sequential model"""
    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = M.Sequential()

        self._add_layers()
        self._compile_model()

    def build(self) -> Model:
        return Model(self.model, self.config)

    def _add_layers(self):
        for layer in self.config['model']['layers']:
            self._add_layer(layer)
        print('[Model] Layers added')

    def _compile_model(self):
        compilation = self.config['model']['compilation']
        loss = compilation['loss']
        optimizer = compilation['optimizer']
        metrics = compilation['metrics']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print('[Model] Model Compiled')

    def _add_layer(self, config_layer: dict):
        layer_type = config_layer['type']
        if layer_type == 'input':
            self._add_input(config_layer)
        elif layer_type == 'dense':
            self._add_dense(config_layer)
        elif layer_type == 'lstm':
            self._add_lstm(config_layer)
        elif layer_type == 'gru':
            self._add_gru(config_layer)
        elif layer_type == 'attention':
            self._add_attention(config_layer)
        elif layer_type == 'dropout':
            self._add_dropout(config_layer)
        elif layer_type == 'conv1d':
            self._add_conv1d(config_layer)
        elif layer_type == 'averagePooling1d':
            self._add_average_pooling_1d(config_layer)
        elif layer_type == 'batchNormalization':
            self._add_batch_normalization()
        elif layer_type == 'flatten':
            self._add_flatten()
        else:
            raise NotImplementedError

    def _add_input(self, config_layer):
        input_timesteps = config_layer['input_timesteps']
        input_dim = config_layer['input_dim']
        layer = L.Input(shape=(input_timesteps, input_dim))
        self.model.add(layer)

    def _add_dense(self, config_layer):
        neurons = config_layer['neurons']
        activation = config_layer['activation'] if 'activation' in config_layer else None
        layer = L.Dense(neurons, activation=activation)
        self.model.add(layer)

    def _add_lstm(self, config_layer):
        neurons = config_layer['neurons']
        return_seq = config_layer['return_seq']
        input_timesteps = config_layer['input_timesteps'] if 'input_timesteps' in config_layer else None
        input_dim = config_layer['input_dim'] if 'input_dim' in config_layer else None
        layer = L.LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq)
        self.model.add(layer)

    def _add_gru(self, config_layer):
        neurons = config_layer['neurons']
        return_seq = config_layer['return_seq']
        input_timesteps = config_layer['input_timesteps'] if 'input_timesteps' in config_layer else None
        input_dim = config_layer['input_dim'] if 'input_dim' in config_layer else None
        layer = L.GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq)
        self.model.add(layer)

    def _add_attention(self, config_layer):
        # TODO Implement Attention Layer
        raise NotImplementedError

    def _add_dropout(self, config_layer):
        dropout_rate = config_layer['rate']
        layer = L.Dropout(dropout_rate)
        self.model.add(layer)

    def _add_flatten(self):
        layer = L.Flatten()
        self.model.add(layer)

    def _add_conv1d(self, config_layer):
        filters = config_layer['filters']
        kernel_size = config_layer['kernel_size']
        activation = config_layer['activation']
        layer = L.Conv1D(filters=filters, kernel_size=kernel_size, activation=activation)
        self.model.add(layer)

    def _add_average_pooling_1d(self, config_layer):
        pool_size = config_layer['pool_size']
        layer = L.AveragePooling1D(pool_size)
        self.model.add(layer)

    def _add_batch_normalization(self):
        layer = L.BatchNormalization()
        self.model.add(layer)
