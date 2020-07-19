import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import LSTM, Dropout

from specusticc.configs_init.model.agent_config import AgentConfig


class GAN:
    def __init__(self, config: AgentConfig):
        self.epochs = 200
        self.batch_size = 500
        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features + config.context_features
        self.output_timesteps = config.output_timesteps

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        real_input = Input(shape=(self.input_timesteps, self.input_features))
        predictions = self.generator(real_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(predictions)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(real_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)



    def build_generator(self):

        model = Sequential()

##########
        model.add(LSTM(units=100, input_shape=(self.input_timesteps, self.input_features), return_sequences=True))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=self.output_timesteps, activation="linear"))

        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.output_timesteps), activation='tanh'))
        # model.add(Reshape(self.output_timesteps))
##########
        real_data = Input(shape=(self.input_timesteps, self.input_features))
        predictions = model(real_data)

        model.summary()

        return Model(real_data, predictions)

    def build_discriminator(self):

        model = Sequential()
##########
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
##########
        predictions = Input(shape=self.output_timesteps)
        validity = model(predictions)

        model.summary()

        return Model(predictions, validity)

    def train(self, X, Y):

        # Adversarial ground truths
        valid = np.ones((len(Y), 1))
        fake = np.zeros((len(Y), 1))

        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Generate a batch of new images
            # gen_imgs = Y_pred
            Y_pred = self.generator.predict(X)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(Y, valid)
            d_loss_fake = self.discriminator.train_on_batch(Y_pred, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(X, valid)

            # Plot the progress
            print(f'{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]')

        return self.generator