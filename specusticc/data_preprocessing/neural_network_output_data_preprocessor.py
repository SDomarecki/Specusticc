from specusticc.configs_init.preprocessor_config import PreprocessorConfig

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class NeuralNetworkOutputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff
        self.features = config.features

    def transform_output(self, data):
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

        scaler = MinMaxScaler(feature_range=(0, 1))
        output_values = scaler.fit_transform(output_values)

        output = output_values.reshape(samples, self.prediction)
        return output, scaler, org_columns, output_dates

