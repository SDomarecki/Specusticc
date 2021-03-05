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
        optimizer = ["Adam"]
        neurons = [20, 100]
        activation = ["relu"]
        dropout_rate = [0.2, 0.8]

        self.possible_parameters = dict(
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            neurons=neurons,
            activation=activation,
        )

    def build_model(
        self, optimizer="adam", dropout_rate=0.0, neurons=100, middle_layers=1
    ):

        encoder_in = L.Input(
            shape=(self.context_timesteps, self.context_features), name="Encoder_input"
        )
        middle_encoder = encoder_in
        for i in range(middle_layers):
            lstm = L.LSTM(neurons, return_sequences=True)
            middle_encoder = lstm(middle_encoder)
            middle_encoder = L.Dropout(dropout_rate)(middle_encoder)
        encoder = L.LSTM(neurons, return_sequences=True, return_state=True)
        encoder_out, state_h, state_c = encoder(middle_encoder)
        encoder_out = L.Dropout(dropout_rate)(encoder_out)
        encoder_states = [state_h, state_c]

        decoder_in = L.Input(
            shape=(self.input_timesteps, self.input_features), name="Decoder_input"
        )
        middle_decoder = decoder_in
        for i in range(middle_layers):
            lstm = L.LSTM(self.input_features, return_sequences=True)
            middle_decoder = lstm(middle_decoder)
            middle_decoder = L.Dropout(dropout_rate)(middle_decoder)
        decoder_lstm = L.LSTM(neurons, return_sequences=True, return_state=True)
        decoder_out, _, _ = decoder_lstm(middle_decoder, initial_state=encoder_states)
        decoder_out = L.GlobalAveragePooling1D()(decoder_out)
        decoder_out = L.Dropout(dropout_rate)(decoder_out)

        attn_out = L.Attention(name="attention_layer")([encoder_out, decoder_out])
        attn_out = L.GlobalAveragePooling1D()(attn_out)
        decoder_concat = L.Concatenate(name="concat_layer")([decoder_out, attn_out])

        # decoder_concat = L.Flatten()(decoder_concat)
        decoder_dense = L.Dense(self.output_timesteps, activation="linear")

        decoder_out = decoder_dense(decoder_concat)

        model = M.Model([encoder_in, decoder_in], decoder_out)

        mape = "mean_absolute_percentage_error"
        model.compile(loss=mape, optimizer=optimizer, metrics=[mape])

        return model
