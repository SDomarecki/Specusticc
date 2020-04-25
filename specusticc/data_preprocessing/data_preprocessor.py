from datetime import datetime, timedelta

import pandas as pd

from specusticc.configs_init.preprocessor_config import PreprocessorConfig
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.data_preprocessing.neural_network_input_data_preprocessor import NeuralNetworkInputDataPreprocessor
from specusticc.data_preprocessing.neural_network_output_data_preprocessor import NeuralNetworkOutputDataPreprocessor
from specusticc.data_preprocessing.tree_data_processor import TreeDataProcessor


class DataPreprocessor:
    def __init__(self, data: dict, config: PreprocessorConfig) -> None:
        self.config = config
        self.input = data['input']
        self.output = data['output']

        self.data_holder = None

    def get_data(self) -> DataHolder:
        self._process_data()
        return self.data_holder

    def _process_data(self):
        self._limit_columns()
        self._filter_by_dates()
        self._reshape_data_to_model_input_output()

    def _limit_columns(self):
        for ticker, df in self.input.items():
            self.input[ticker] = df[self.config.input_columns]
        for ticker, df in self.output.items():
            self.output[ticker] = df[self.config.output_columns]

    def _filter_by_dates(self):
        for ticker, df in self.input.items():
            self.train_input = _filter_history_by_dates(df, self.config.train_date)
            self.test_input = _filter_history_by_dates(df, self.config.test_date)
        for ticker, df in self.output.items():
            self.train_output = _filter_history_by_dates(df, self.config.train_date)
            self.test_output = _filter_history_by_dates(df, self.config.test_date)

    def _reshape_data_to_model_input_output(self):
        model_type = self.config.model_type
        if model_type == 'decision_tree':
            self._reshape_data_to_tree()
            #TODO complete Tree model
        elif model_type == 'neural_network':
            self._reshape_data_to_neural_network()
        else:
            raise NotImplementedError

    def _reshape_data_to_tree(self):
        dh = DataHolder()
        data2i = TreeDataProcessor(self.config)
        data2o = TreeDataProcessor(self.config)
        # TODO complete Tree model

    def _reshape_data_to_neural_network(self):
        dh = DataHolder()
        data2i = NeuralNetworkInputDataPreprocessor(self.config)
        data2o = NeuralNetworkOutputDataPreprocessor(self.config)

        dh.train_input, dh.train_input_scaler = data2i.transform_input(self.train_input)
        dh.train_output, dh.train_output_scaler = data2o.transform_output(self.train_output)
        dh.test_input, dh.test_input_scaler = data2i.transform_input(self.test_input)
        dh.test_output, dh.test_output_scaler = data2o.transform_output(self.test_output)

        self.data_holder = dh


def _filter_history_by_dates(df: pd.DataFrame, dates: dict) -> pd.DataFrame:
    if dates is None:
        return df

    from_date = dates['from']
    to_date = dates['to']
    from_index = _get_closest_date_index(df, from_date)
    to_index = _get_closest_date_index(df, to_date)
    filtered = df.iloc[from_index:to_index]
    return filtered


def _get_closest_date_index(df: pd.DataFrame, date: datetime) -> int:
    closest_index = None
    while closest_index is None:
        try:
            closest_index = df.index[df['date'] == date][0]
        except Exception:
            pass
        date = date - timedelta(days=1)
    return closest_index