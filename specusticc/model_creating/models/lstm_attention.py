import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model.agent_config import AgentConfig


class LSTMAttention:
    def __init__(self, config: AgentConfig):
        self.epochs = 50

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
                    optimizer='adam',
                    dropout_rate=0.0,
                    neurons=20,
                    activation='relu'):
        # define training encoder
        encoder_in = L.Input(shape=(self.context_timesteps, self.context_features))
        encoder = L.LSTM(100, return_sequences=True, return_state=True)
        encoder_out, state_h, state_c = encoder(encoder_in)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_in = L.Input(shape=(self.input_timesteps, self.input_features))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = L.LSTM(100, return_sequences=True, return_state=True)
        decoder_out, _, _ = decoder_lstm(decoder_in, initial_state=encoder_states)
        decoder_out = L.GlobalAveragePooling1D()(decoder_out)

        attn_out = L.Attention(name='attention_layer')([encoder_out, decoder_out])
        attn_out = L.GlobalAveragePooling1D()(attn_out)
        decoder_concat = L.Concatenate(name='concat_layer')([decoder_out, attn_out])
        decoder_concat = L.Flatten()(decoder_concat)
        decoder_dense = L.Dense(self.output_timesteps, activation='linear')

        decoder_out = decoder_dense(decoder_concat)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = M.Model([encoder_in, decoder_in], decoder_out)

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model