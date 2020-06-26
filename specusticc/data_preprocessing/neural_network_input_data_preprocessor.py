from specusticc.configs_init.preprocessor_config import PreprocessorConfig

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class NeuralNetworkInputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff

    def transform_input(self, data, context=False):
        data = self._reshape_dict_to_unified_dataframe(data)
        return self._reshape_to_numpy_samples(data, context)

    def _reshape_dict_to_unified_dataframe(self, data: dict) -> pd.DataFrame:
        unified_df = pd.DataFrame(columns=['date'])
        for ticker, df in data.items():
            df.columns = [ticker + '_' + str(col) for col in df.columns if col != 'date'] + ['date']
            unified_df = unified_df.merge(df, left_on='date', right_on='date', how='outer')
        unified_df = unified_df.sort_values(by=['date'])\
            .interpolate()\
            .fillna(0.0)
        return unified_df

    def _reshape_to_numpy_samples(self, data: pd.DataFrame, context):
        org_columns = data.columns
        if not context:
            self.org_date = data['date']
        else:
            data = data[data['date'].isin(self.org_date)]

        data = data.drop(columns='date')
        features = len(data.columns)
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
            data_np = data_np.reshape(samples, self.timestamps * features)
        else:
            data_np = data_np.reshape(samples, self.timestamps, features)

        return data_np, scaler, org_columns, self.org_date
