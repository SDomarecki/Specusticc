import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork


class LSTMEncoderDecoder(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        self.context_timesteps = config.context_timesteps
        self.context_features = config.context_features

        super().__init__(config)
        self.epochs = 50
        self.batch_size = 500

    def _build_model(self) -> None:
        # define training encoder
        encoder_inputs = L.Input(shape=(self.context_timesteps, self.context_features))
        encoder = L.LSTM(100, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = L.Input(shape=(self.input_timesteps, self.input_features))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = L.LSTM(100, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_outputs = L.Flatten()(decoder_outputs)
        decoder_dense = L.Dense(self.output_timesteps, activation='linear')

        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.predictive_model = M.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
