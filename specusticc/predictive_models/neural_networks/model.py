import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.models import Sequential

from specusticc.utilities.timer import Timer


class Model:
    def __init__(self, model: Sequential, config: dict) -> None:
        self.model = model
        self.epochs = config['model']['fitting']['epochs']
        self.batch_size = config['model']['fitting']['batch_size']

    def save(self, save_dir: str) -> None:
        self.model.save(save_dir)

    def train(self, x: np.array, y: np.array) -> None:
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.epochs, self.batch_size))
        t = Timer()
        t.start()

        save_fname = 'temp.h5'
        callbacks = [
            ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.01, patience=3, verbose=1),
            ModelCheckpoint(filepath=save_fname, monitor='categorical_accuracy', save_best_only=True, verbose=1),
            TensorBoard(),
            CSVLogger(filename='learning.log')
        ]
        self.model.fit(
            x,
            y,
            epochs=self.epochs,
            callbacks=callbacks
        )
        self.model.save(save_fname)
        print('[Model] Training Completed')
        t.stop()
        t.print_time()

    def predict_classification(self, test_data:np.array) -> []:
        print('[Model] Predicting position classes...')
        predictions = self.model.predict(test_data)
        return predictions

    def predict_regression(self, test_data: np.array) -> []:
        raise NotImplementedError

