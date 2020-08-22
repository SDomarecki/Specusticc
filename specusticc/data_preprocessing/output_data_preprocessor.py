import numpy as np
import pandas as pd

from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig


class OutputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff

    def transform_output(self, data: pd.DataFrame):
        org_columns = data.columns
        org_date = data[['date']]

        data = data.drop(columns='date')
        output_values = pd.DataFrame()
        output_dates = pd.DataFrame(columns=['date'])
        samples = int((len(data) - self.prediction - self.timestamps) / self.sample_time_diff)
        for i in range(samples):
            start_index = i * self.sample_time_diff + self.timestamps
            end_index = start_index + self.prediction
            output_values = output_values.append(data.iloc[start_index: end_index])
            output_dates = output_dates.append(org_date.iloc[start_index: end_index])
        output_values = output_values.to_numpy()

        output = output_values.reshape(samples, self.prediction)
        output, scaler = self.detrend(output)

        return output, scaler, org_columns, output_dates

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