import datetime as dt
import os

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import newaxis


class Model:
    def __init__(self, model: Sequential) -> None:
        self.model = model

    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        scaled_x = self.scale(x)
        save_fname = os.path.join(save_dir, 'neural_network-%s-e%s.h5' % (dt.datetime.now().strftime('%Y-%m-%d_%H-%M'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit(
            scaled_x,
            y,
            epochs=epochs,
            callbacks=callbacks
        )
        self.model.save(save_fname)
        print('[Model] Training Completed. Model saved as %s' % save_fname)

    def predict_classification(self, test_data:np.array) -> []:
        print('[Model] Predicting position classes...')
        scaled_data = self.scale(test_data)
        predictions = self.model.predict(scaled_data)
        return predictions

    def predict_sequences_multiple(self, test_data: np.array, window_size, prediction_len):
        print('[Model] Predicting Sequences Multiple...')
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

    def scale(self, data: np.array) -> np.array:
        from sklearn.preprocessing import StandardScaler
        org_shape = data.shape
        temp_shape = (data.shape[0]*data.shape[1], data.shape[2])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.reshape(data, temp_shape))
        reshaped = np.reshape(scaled, org_shape)
        return reshaped

