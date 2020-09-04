import numpy as np
import pandas as pd

from configs_init.model.preprocessor_config import PreprocessorConfig


class InputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.timestamps = config.window_length
        self.prediction = config.horizon
        self.rolling = config.rolling

    def transform_input(self, data: pd.DataFrame):
        data = data.drop(columns='date')
        input_samples = pd.DataFrame()
        features = len(data.columns)
        samples = int((len(data) - self.prediction - self.timestamps) / self.rolling)
        for i in range(samples):
            start_index = i * self.rolling
            end_index = start_index + self.timestamps
            input_samples = input_samples.append(data.iloc[start_index: end_index])

        input_samples = input_samples.to_numpy()

        output = input_samples.reshape(samples, self.timestamps, features)
        output, scaler = self.detrend(output)

        return output

    def detrend(self, samples):
        scaler = np.array([]) # (samples, features)
        detrended_samples = np.empty(samples.shape)
        for i in range(len(samples)):
            sample = samples[i]
            scaler = np.append(scaler, sample[0])
            detrended_sample = np.ones(sample.shape)
            for j in range(1, len(sample)):
                detrended_sample[j] = sample[j]/sample[0]
            detrended_samples[i] = detrended_sample
        return detrended_samples, scaler