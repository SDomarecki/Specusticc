import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.model_creating.models.transformer_classes.decoder import Decoder
from specusticc.model_creating.models.transformer_classes.encoder import Encoder

from specusticc.model_creating.neural_network import NeuralNetwork


# see: Attention Is All You Need, url:https://arxiv.org/abs/1706.03762
class ModelTransformer(NeuralNetwork):
    def __init__(self, config: ModelCreatorConfig):
        self.context_timesteps = config.context_timesteps
        self.context_features = config.context_features

        super().__init__(config)

        self.epochs = 50
        self.batch_size = 500

    def _build_model(self) -> None:
        H = 2
        NUM_LAYERS = 2
        MODEL_SIZE = self.input_timesteps

        encoder_inputs = L.Input(shape=(self.context_timesteps, self.context_features))
        encoder = Encoder(MODEL_SIZE, NUM_LAYERS, H)
        encoder_outputs = encoder(encoder_inputs)

        decoder_inputs = L.Input(shape=(self.input_timesteps, self.input_features))
        decoder = Decoder(self.output_timesteps, MODEL_SIZE, NUM_LAYERS, H)
        decoder_outputs = decoder(decoder_inputs, encoder_outputs)

        self.predictive_model = M.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def _compile_model(self) -> None:
        self.predictive_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
