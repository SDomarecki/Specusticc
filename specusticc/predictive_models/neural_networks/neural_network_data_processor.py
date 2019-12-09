import numpy as np
import pandas as pd

from sklearn import preprocessing

from specusticc.data_processing.ratio_to_class import ratio_to_class


class NeuralNetworkDataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config

        self.samples = None
        self.timestamps = config['preprocessing']['sequence_length']
        self.features = 0

        self.input = None
        self.output = None
        self.input_len = None

        self._count_features()

    def transform_for_model(self, input_data: dict, output_data: dict) -> tuple:
        input_dataframe = self._unpack_data(input_data)
        output_dataframe = self._unpack_data(output_data)

        self._count_samples(input_dataframe)

        target = self.config['model']['target']
        if target == 'regression':
            self._reshape_for_regression(input_dataframe)
            self._shift_for_regression(output_dataframe)
        elif target == 'classification':
            self._calculate_classes(output_dataframe)
            self._reshape_for_classification(input_dataframe)
        else:
            raise NotImplementedError
        return self.get_io()

    def _align_to_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        seq_len = self.config['pre']['sequence_length']
        shift = self.config['data']['sequence_shift']

        len_data = len(data)
        self.input_len = int((len_data - seq_len) / shift)
        last_input_index = self.input_len * seq_len
        return data.iloc[:last_input_index]

    def _reshape_for_classification(self, input_data: pd.DataFrame) -> None:
        sample_diff = self.config['preprocessing']['sample_time_difference']
        input_data_normalized = self._normalize(input_data)
        data_np = []
        for i in range(self.samples):
            start_index = i * sample_diff
            end_index = start_index + self.timestamps
            data_np.append(input_data_normalized.iloc[start_index: end_index])
        data_np = pd.concat(data_np).to_numpy()

        first_layer_type = self.config['model']['layers'][0]['type']
        if first_layer_type == 'dense':
            data_np = data_np.reshape(self.samples, self.timestamps * self.features)
        else:
            data_np = data_np.reshape(self.samples, self.timestamps, self.features)
        self.input = data_np

    def _normalize(self, input_data: pd.DataFrame) -> pd.DataFrame:
        val = input_data.values
        scaler = preprocessing.StandardScaler()
        val_scaled = scaler.fit_transform(val)
        input_data_normalized = pd.DataFrame(val_scaled, columns=input_data.columns)
        return input_data_normalized

    def _calculate_classes(self, data: pd.DataFrame) -> None:
        sequence_prediction_time = self.config['preprocessing']['sequence_prediction_time']
        sample_time_difference = self.config['preprocessing']['sample_time_difference']

        output_labels = []
        for i in range(self.samples - 1):
            present_val_index = i * sample_time_difference + self.timestamps - 1
            future_val_index = present_val_index + sequence_prediction_time

            # TODO change this ugly hardcode asap
            present_val = data.loc[:, 'PKO_close'].iloc[present_val_index]
            future_val = data.loc[:, 'PKO_close'].iloc[future_val_index]
            ratio = future_val / present_val

            ratio_class = ratio_to_class(ratio)
            output_labels.append(ratio_class)

        ratio_class = ratio_to_class(1.)
        output_labels.append(ratio_class)
        self.output = np.array(output_labels)

    def get_io(self) -> tuple:
        return self.input, self.output

    def _count_features(self):
        inputs = self.config['import']['input']
        for input in inputs:
            columns = input['columns']
            self.features += len(columns) - 1

    def _count_samples(self, data: pd.DataFrame) -> None:
        prediction = self.config['preprocessing']['sequence_prediction_time']
        sample_time_diff = self.config['preprocessing']['sample_time_difference']

        self.samples = int((len(data) - prediction - self.timestamps)/sample_time_diff)

    def _reshape_for_regression(self, input_data):
        raise NotImplementedError

    def _shift_for_regression(self, output_data):
        raise NotImplementedError

    def _unpack_data(self, data: dict) -> pd.DataFrame:
        history_array = []
        tickers = []
        for ticker, history in data.items():
            new_history = history.copy()
            new_history.columns = ticker + '_' + new_history.columns
            history_array.append(new_history)
            tickers.append(ticker)

        merged = history_array[0]
        first_prefix_date = tickers[0] + '_date'
        for i in range(1, len(history_array)):
            next_prefix_date = tickers[i] + '_date'
            merged = pd.merge(merged, history_array[i],
                              left_on=first_prefix_date,
                              right_on=next_prefix_date)
            merged = merged.drop(columns=[next_prefix_date])
        merged = merged.drop(columns=[first_prefix_date])
        return merged
