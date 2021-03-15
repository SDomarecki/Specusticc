import tensorflow as tf

from specusticc.model_creating.models.transformer_classes.decoder_layer import (
    DecoderLayer,
)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_size, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"num_layers": self.num_layers})
        return config

    def call(self, inputs, **kwargs):
        [x, enc_output] = inputs
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i]([x, enc_output])

        # x.shape == (batch_size, target_seq_len, d_model)
        x = self.flatten(x)
        x = self.dense(x)
        return x
