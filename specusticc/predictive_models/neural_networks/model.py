import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.models import Sequential

from specusticc.data_processing.data_holder import DataHolder
from specusticc.utilities.timer import Timer

import matplotlib.pyplot as plt


class Model:
    def __init__(self, model: Sequential, config: dict) -> None:
        self.model = model
        self.epochs = config['model']['fitting']['epochs']
        self.batch_size = config['model']['fitting']['batch_size']

    def save(self, save_dir: str) -> None:
        self.model.save(save_dir)

    def train(self, data: DataHolder) -> None:
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.epochs, self.batch_size))
        t = Timer()
        t.start()

        x_train = data.train_input
        y_train = data.train_output

        save_fname = 'temp.h5'
        callbacks = [
            ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.01, patience=3, verbose=1),
            ModelCheckpoint(filepath=save_fname, monitor='categorical_accuracy', save_best_only=True, verbose=1),
            TensorBoard(),
            CSVLogger(filename='learning.log')
        ]
        history = self.model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=self.epochs,
            callbacks=callbacks
        )
        self.model.save(save_fname)
        print('[Model] Training Completed')
        t.stop()
        t.print_time()

        self._plot_history(history)

    def _plot_history(self, history):
        # summarize history for accuracy
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def predict(self, test_data: np.array) -> []:
        print('[Model] Predicting...')
        predictions = self.model.predict(test_data)
        return predictions
