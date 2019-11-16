import datetime as dt
import os

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numpy import newaxis


class Model:
    def __init__(self, model: Sequential, config: dict) -> None:
        self.model = model
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.save_dir = config['model']['save_dir']
        self.seq_len = config['data']['sequence_length']

    def save(self, save_dir: str) -> None:
        self.model.save(save_dir)

    def train(self, x: np.array, y: np.array) -> None:
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.epochs, self.batch_size))

        scaled_x = self._scale(x)
        save_fname = os.path.join('temp.h5')
        callbacks = [
            ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.01, patience=3, verbose=1),
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit(
            scaled_x,
            y,
            epochs=self.epochs,
            callbacks=callbacks
        )
        self.model.save(save_fname)
        print('[Model] Training Completed')

    def predict_classification(self, test_data:np.array) -> []:
        print('[Model] Predicting position classes...')
        scaled_data = self._scale(test_data)
        predictions = self.model.predict(scaled_data)
        return predictions

    def predict_sequences_multiple(self, test_data: np.array) -> []:
        print('[Model] Predicting Sequences Multiple...')
        window_size = self.seq_len
        prediction_len = self.seq_len

        prediction_seqs = []
        for i in range(int(len(test_data) / prediction_len)):
            curr_frame = test_data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def _scale(self, data: np.array) -> np.array:
        from sklearn.preprocessing import StandardScaler
        org_shape = data.shape
        temp_shape = (data.shape[0]*data.shape[1], data.shape[2])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.reshape(data, temp_shape))
        reshaped = np.reshape(scaled, org_shape)
        return reshaped
