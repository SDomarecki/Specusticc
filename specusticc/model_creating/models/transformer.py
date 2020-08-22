import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O

from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.model_creating.models.transformer_classes.decoder import Decoder
from specusticc.model_creating.models.transformer_classes.encoder import Encoder


# see: Attention Is All You Need, url:https://arxiv.org/abs/1706.03762
class ModelTransformer:
    def __init__(self, config: AgentConfig):
        self.epochs = 60

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features
        self.output_timesteps = config.output_timesteps

        self.context_timesteps = config.context_timesteps
        self.context_features = config.context_features

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
                    dropout_rate=0.01,
                    neurons=20,
                    num_stacks=4):

        # Do opisania co to jest
        #d_model
        #num_heads
        #dff

        d_model = self.context_features
        num_heads = self.context_features
        dff = self.context_features

        encoder_inputs = L.Input(shape=(self.context_timesteps, self.context_features))
        encoder = Encoder(num_stacks, d_model, num_heads, dff, rate=dropout_rate)
        encoder_outputs = encoder(encoder_inputs)

        d_model = num_heads = dff = self.input_features
        decoder_inputs = L.Input(shape=(self.input_timesteps, self.input_features))
        decoder = Decoder(self.output_timesteps, num_stacks, d_model, num_heads, dff, rate=dropout_rate)
        dec_output = decoder(decoder_inputs, encoder_outputs)

        model = M.Model([encoder_inputs, decoder_inputs], dec_output)

        opt = O.Adam(learning_rate=0.05)
        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=opt, metrics=[mape])

        return model
