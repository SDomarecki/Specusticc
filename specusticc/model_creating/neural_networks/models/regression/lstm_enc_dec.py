import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.neural_networks.neural_network import NeuralNetwork

import numpy as np


class LSTMEncoderDecoder(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        super().__init__(config)
        self.epochs = 50
        self.batch_size = 500

    def _build_model(self) -> None:
        # Define an input sequence and process it.
        encoder_inputs = L.Input(shape=(self.input_timesteps, self.input_features))
        encoder = L.LSTM(100, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = L.Input(shape=(self.output_timesteps, self.input_features))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = L.LSTM(100, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = L.Dense(1, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.predictive_model = M.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # # define inference encoder
        # self.encoder_model = M.Model(encoder_inputs, encoder_states)
        # # define inference decoder
        # decoder_state_input_h = L.Input(shape=(self.input_timesteps, self.input_features))
        # decoder_state_input_c = L.Input(shape=(self.input_timesteps, self.input_features))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        # decoder_states = [state_h, state_c]
        # decoder_outputs = decoder_dense(decoder_outputs)
        # self.decoder_model = M.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def _build_model_obs(self) -> None:
        self.predictive_model = M.Sequential()

        self.predictive_model.add(L.LSTM(units=100, input_shape=(self.input_timesteps, self.input_features), return_sequences=True))
        self.predictive_model.add(L.LSTM(units=100, return_sequences=True))
        self.predictive_model.add(L.Dropout(rate=0.2))
        self.predictive_model.add(L.LSTM(units=100, return_sequences=False))
        self.predictive_model.add(L.Dropout(rate=0.2))
        self.predictive_model.add(L.Dense(units=self.output_timesteps, activation="linear"))

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

    def predict(self, test_data: np.array) -> []:
        print('[Model] Predicting...')
        predictions = self.predictive_model.predict([test_data, np.zeros_like(test_data)])
        return predictions
    # def predict(self, test_data: np.array) -> []:
    #     for sample in test_data:
    #         # encode
    #         state = self.encoder_model.predict(sample)
    #         # start of sequence input
    #         target_seq = []
    #         # collect predictions
    #         output = list()
    #         for t in range(len(test_data)):
    #             # predict next char
    #             yhat, h, c = self.decoder_model.predict([target_seq] + state)
    #             # store prediction
    #             output.append(yhat[0, 0, :])
    #             # update state
    #             state = [h, c]
    #             # update target sequence
    #             target_seq = yhat
    #     return output