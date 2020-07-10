import tensorflow as tf

from specusticc.model_creating.models.transformer_classes.decoder_layer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_size, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        # self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_size)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers
        })
        return config

    def call(self, x, enc_output):

        # seq_len = tf.shape(x)[1]
        # attention_weights = {}

        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output)

            # attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            # attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        x = self.flatten(x)
        x = self.dense(x)
        return x