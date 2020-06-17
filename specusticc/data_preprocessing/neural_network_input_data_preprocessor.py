from specusticc.configs_init.preprocessor_config import PreprocessorConfig

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class NeuralNetworkInputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff
        self.features = config.features

    def transform_input(self, data):
        org_columns = data.columns
        org_date = data['date']

        data = data.drop(columns='date')
        data_np = []
        samples = int((len(data) - self.prediction - self.timestamps) / self.sample_time_diff)
        for i in range(samples):
            start_index = i * self.sample_time_diff
            end_index = start_index + self.timestamps
            data_np.append(data.iloc[start_index: end_index])

        data_np = pd.concat(data_np).to_numpy()

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_np = scaler.fit_transform(data_np)

        if self.config.input_dim == 2:
            data_np = data_np.reshape(samples, self.timestamps * self.features)
        else:
            data_np = data_np.reshape(samples, self.timestamps, self.features)

        return data_np, scaler, org_columns, org_date
