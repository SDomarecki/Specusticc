from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class InputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff

    def transform_input(self, data: pd.DataFrame):
        org_columns = data.columns
        org_date = data[['date']]

        data = data.drop(columns='date')
        input_samples = pd.DataFrame()
        features = len(data.columns)
        samples = int((len(data) - self.prediction - self.timestamps) / self.sample_time_diff)
        for i in range(samples):
            start_index = i * self.sample_time_diff
            end_index = start_index + self.timestamps
            input_samples = input_samples.append(data.iloc[start_index: end_index])

        input_samples = input_samples.to_numpy()

        scaler = MinMaxScaler(feature_range=(0, 1))
        input_samples = scaler.fit_transform(input_samples)

        output = input_samples.reshape(samples, self.timestamps, features)
        return output
