import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from configs_init.model.agent_config import AgentConfig


class LSTMEncoderDecoder:
    def __init__(self, config: AgentConfig):
        self.epochs = 100

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
        dropout_rate = [0.0, 0.2, 0.8]
        middle_layers = [0, 1, 2, 4]

        self.possible_parameters = dict(
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                middle_layers=middle_layers)

    def build_model(self,
                    optimizer='adam',
                    middle_layers=1,
                    dropout_rate=0.2):
        # define training encoder
        encoder_inputs = L.Input(shape=(self.context_timesteps, self.context_features), name='Encoder_input')
        middle_encoder = encoder_inputs
        for i in range(middle_layers):
            lstm = L.LSTM(self.context_features, return_sequences=True)
            middle_encoder = lstm(middle_encoder)
            middle_encoder = L.Dropout(dropout_rate)(middle_encoder)
        encoder = L.LSTM(self.context_features, return_state=True, name='Encoder_LSTM')
        encoder_outputs, state_h, state_c = encoder(middle_encoder)
        encoder_states = [state_h, state_c]


        decoder_inputs = L.Input(shape=(self.input_timesteps, self.input_features), name='Decoder_input')
        middle_decoder = decoder_inputs
        for i in range(middle_layers):
            lstm = L.LSTM(self.input_features, return_sequences=True)
            middle_decoder = lstm(middle_decoder)
            middle_decoder = L.Dropout(dropout_rate)(middle_decoder)

        decoder_lstm = L.LSTM(self.context_features, return_sequences=True, name='Decoder_LSTM')
        decoder_outputs = decoder_lstm(middle_decoder, initial_state=encoder_states)
        decoder_outputs = L.Flatten()(decoder_outputs)
        decoder_dense = L.Dense(self.output_timesteps, activation='linear', name='Dense_output')

        decoder_outputs = decoder_dense(decoder_outputs)

        model = M.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model
