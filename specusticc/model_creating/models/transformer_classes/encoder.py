import tensorflow as tf

from specusticc.model_creating.models.transformer_classes.encoder_layer import EncoderLayer


class Encoder(tf.keras.models.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers
        })
        return config

    def call(self, x):
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)