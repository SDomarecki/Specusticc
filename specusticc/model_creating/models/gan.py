import numpy as np
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import matplotlib.pyplot as plt

from specusticc.configs_init.model.agent_config import AgentConfig


class GAN:
    def __init__(self, config: AgentConfig, epochs: int):
        self.epochs = epochs
        self.batch_size = 50
        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features + config.context_features
        self.output_timesteps = config.output_timesteps

        G_optimizer = O.Adam(learning_rate=0.0002)
        D_optimizer = O.Adam(learning_rate=0.0001)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=D_optimizer,
            metrics=['accuracy'])

        mape = 'mean_absolute_percentage_error'
        self.generator = self.build_generator()
        self.generator.compile(loss=mape,
            optimizer=G_optimizer,
            metrics=[mape])

        # The generator takes noise as input and generates imgs
        real_input = L.Input(shape=(self.input_timesteps, self.input_features))
        predictions = self.generator(real_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(predictions)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = M.Model(real_input, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=D_optimizer,
                              metrics=['accuracy'])

    def build_generator(self):
        dropout_rate = 0.0
        G = M.Sequential(name='Generator')
        G.add(L.Input(shape=(self.input_timesteps, self.input_features)))
        G.add(L.LSTM(units=20, return_sequences=True))
        G.add(L.Dropout(dropout_rate))

        G.add(L.LSTM(units=10, return_sequences=True))
        G.add(L.Dropout(dropout_rate))

        G.add(L.LSTM(units=5, return_sequences=True))
        G.add(L.Dropout(dropout_rate))

        G.add(L.Flatten())

        G.add(L.Dense(units=200, activation="relu"))
        G.add(L.Dense(units=100, activation="relu"))
        G.add(L.Dense(units=self.output_timesteps, activation="linear"))

        G.summary()

        return G

    def build_discriminator(self):
        dropout_rate = 0.0
        activation = 'relu'

        D = M.Sequential(name='Discriminator')
        D.add(L.Input(shape=self.output_timesteps))
        D.add(L.Reshape(target_shape=(self.output_timesteps, 1)))

        D.add(L.Conv1D(filters=64, kernel_size=2, activation=activation))
        D.add(L.Conv1D(filters=64, kernel_size=2, activation=activation))
        D.add(L.AveragePooling1D(pool_size=2))
        D.add(L.Dropout(rate=dropout_rate))


        D.add(L.Flatten())
        D.add(L.Dense(units=200, activation="relu"))
        D.add(L.Dense(units=20, activation="relu"))
        D.add(L.Dense(1, activation='sigmoid'))

        D.summary()

        return D

    def train(self, X, Y):
        # Adversarial ground truths
        valid = np.ones((len(Y), 1))
        fake = np.zeros((len(Y), 1))

        accuracy_array = []
        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            Y_pred = self.generator.predict(X)
            self.generator.evaluate(X, Y)

            d_loss_real = self.discriminator.train_on_batch(Y, valid)
            d_loss_fake = self.discriminator.train_on_batch(Y_pred, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(X, valid)

            print(f'{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]')
            accuracy_array.append(100*d_loss[1])

        plt.plot(accuracy_array)
        plt.title('Discriminator accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.grid()
        plt.show()

        return self.generator