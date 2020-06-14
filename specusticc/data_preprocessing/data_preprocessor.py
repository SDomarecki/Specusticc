from datetime import datetime, timedelta

import pandas as pd

from specusticc.configs_init.preprocessor_config import PreprocessorConfig
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.data_preprocessing.neural_network_input_data_preprocessor import NeuralNetworkInputDataPreprocessor
from specusticc.data_preprocessing.neural_network_output_data_preprocessor import NeuralNetworkOutputDataPreprocessor


class DataPreprocessor:
    def __init__(self, data: dict, config: PreprocessorConfig) -> None:
        self.config = config
        self.input = data['input']
        self.output = data['output']
        self.context = data['context']

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
        if self.context:
            for ticker, df in self.context.items():
                self.context[ticker] = df[self.config.context_columns]

    def _filter_by_dates(self):
        for ticker, df in self.input.items():
            self.train_input = _filter_history_by_dates(df, self.config.train_date)
            self.test_input = _filter_history_by_dates(df, self.config.test_date)
        for ticker, df in self.output.items():
            self.train_output = _filter_history_by_dates(df, self.config.train_date)
            self.test_output = _filter_history_by_dates(df, self.config.test_date)
        if self.context:
            for ticker, df in self.context.items():
                self.train_context = _filter_history_by_dates(df, self.config.train_date)
                self.test_context = _filter_history_by_dates(df, self.config.test_date)

    def _reshape_data_to_model_input_output(self):
        dh = DataHolder()
        data2i = NeuralNetworkInputDataPreprocessor(self.config)
        data2o = NeuralNetworkOutputDataPreprocessor(self.config)

        dh.train_input, dh.train_input_scaler = data2i.transform_input(self.train_input)
        dh.train_output, dh.train_output_scaler = data2o.transform_output(self.train_output)
        dh.test_input, dh.test_input_scaler = data2i.transform_input(self.test_input)
        dh.test_output, dh.test_output_scaler = data2o.transform_output(self.test_output)

        if self.context:
            dh.train_context, dh.train_context_scaler = data2i.transform_input(self.train_context)
            dh.test_context, dh.test_context_scaler = data2i.transform_input(self.test_context)

        self.data_holder = dh


def _filter_history_by_dates(df: pd.DataFrame, dates: dict) -> pd.DataFrame:
    if dates is None:
        return df

    from_date = dates['from']
    to_date = dates['to']

    # TODO funkcja sprawdzająca czy daty from_date, to_date leżą w zakresie df
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
