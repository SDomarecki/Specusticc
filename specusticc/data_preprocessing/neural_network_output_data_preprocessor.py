from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class NeuralNetworkOutputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff
        self.features = config.features

    def transform_output(self, data):
        data = self._reshape_dict_to_unified_dataframe(data)
        return self._reshape_to_numpy_samples(data)

    def _reshape_dict_to_unified_dataframe(self, data: dict) -> pd.DataFrame:
        unified_df = pd.DataFrame()
        for ticker, df in data.items():
            df.columns = [ticker + '_' + str(col) for col in df.columns if col != 'date'] + ['date']
            unified_df = pd.concat([unified_df, df], axis=1)
        unified_df.sort_values(by=['date'])
        return unified_df

    def _reshape_to_numpy_samples(self, data: pd.DataFrame):
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

