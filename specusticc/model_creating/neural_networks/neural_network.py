import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.callbacks as C

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.utilities.timer import Timer


class NeuralNetwork:
    def __init__(self, config: ModelCreatorConfig):
        self.predictive_model = None
        self.callbacks = None
        self.epochs = 0
        self.batch_size = 0
        self.save_fname = 'temp.h5'

        self.input_timesteps = config.nn_input_timesteps
        self.input_features = config.nn_input_features
        self.output_timesteps = config.nn_output_timesteps

        self._build_model()
        self._compile_model()
        self._set_callbacks()

    def save(self, save_dir: str) -> None:
        self.predictive_model.save(save_dir)

    def train(self, data: DataHolder) -> None:
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.epochs, self.batch_size))
        t = Timer()
        t.start()

        x_train = data.train_input
        y_train = data.train_output

        self.predictive_model.summary()
        history = self.predictive_model.fit(
            [x_train, y_train],
            y_train,
            epochs=self.epochs,
            callbacks=self.callbacks
        )
        self.save(self.save_fname)
        print('[Model] Training Completed')
        t.stop()
        t.print_time()

        self._plot_history(history)

    def _plot_history(self, history):
        # summarize history for accuracy
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('mean squared error')
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid()
        plt.show()

    def predict(self, test_data: np.array) -> []:
        print('[Model] Predicting...')
        predictions = self.predictive_model.predict(test_data)
        return predictions

    def _build_model(self) -> None:
        raise NotImplementedError

    def _compile_model(self) -> None:
        raise NotImplementedError

    def _set_callbacks(self) -> None:
        self.callbacks = [
            C.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20),
            C.ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.01, patience=5, verbose=1),
            C.ModelCheckpoint(filepath=self.save_fname, monitor='loss', save_best_only=True, verbose=1),
            # C.TensorBoard(),
            # C.CSVLogger(filename='learning.log')
        ]
