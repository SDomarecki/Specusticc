from datetime import datetime, timedelta

import pandas as pd

from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig
from specusticc.data_loading.loaded_data import LoadedData
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.data_preprocessing.input_data_preprocessor import InputDataPreprocessor
from specusticc.data_preprocessing.output_data_preprocessor import OutputDataPreprocessor


class DataPreprocessor:
    def __init__(self, data: LoadedData, config: PreprocessorConfig):
        self.config = config
        self.input: {} = data.input
        self.output: {} = data.output
        self.context: {} = data.context

        self.input_df: pd.DataFrame = None
        self.output_df: pd.DataFrame = None
        self.context_df: pd.DataFrame = None

        self.data_holder: DataHolder

    def get_data(self) -> DataHolder:
        return self.data_holder

    def preprocess_data(self):
        self._limit_columns()
        self._merge_data_dicts()
        self._filter_by_dates()
        self._limit_context_dates_by_input_dates()
        self._reshape_data_to_neural_network()

    def _limit_columns(self):
        for ticker, df in self.input.items():
            self.input[ticker] = df[self.config.input_columns]
        for ticker, df in self.output.items():
            self.output[ticker] = df[self.config.output_columns]
        for ticker, df in self.context.items():
            self.context[ticker] = df[self.config.context_columns]

    def _merge_data_dicts(self):
        self.input_df = self._dict_to_merged_dataframe(self.input)
        self.output_df = self._dict_to_merged_dataframe(self.output)
        self.context_df = self._dict_to_merged_dataframe(self.context)

    def _dict_to_merged_dataframe(self, data: dict) -> pd.DataFrame:
        unified_df = pd.DataFrame(columns=['date'])
        for ticker, df in data.items():
            df.columns = [ticker + '_' + str(col) for col in df.columns if col != 'date'] + ['date']
            unified_df = unified_df.merge(df, left_on='date', right_on='date', how='outer')
        unified_df = unified_df.sort_values(by=['date'])\
            .interpolate()\
            .fillna(0.0)\
            .reset_index(drop=True)
        return unified_df

    def _filter_by_dates(self):
        self.train_input = _filter_history_by_dates(self.input_df, self.config.train_date)
        self.test_input = _filter_history_by_dates(self.input_df, self.config.test_date)
        self.train_output = _filter_history_by_dates(self.output_df, self.config.train_date)
        self.test_output = _filter_history_by_dates(self.output_df, self.config.test_date)
        self.train_context = _filter_history_by_dates(self.context_df, self.config.train_date)
        self.test_context = _filter_history_by_dates(self.context_df, self.config.test_date)

    def _limit_context_dates_by_input_dates(self):
        self.train_context = self.train_context[self.train_context['date'].isin(self.train_input['date'])]
        self.test_context = self.test_context[self.test_context['date'].isin(self.test_input['date'])]

    def _reshape_data_to_neural_network(self):
        dh = DataHolder()
        data2i_train = InputDataPreprocessor(self.config)
        data2i_test = InputDataPreprocessor(self.config)
        data2o_train = OutputDataPreprocessor(self.config)
        data2o_test = OutputDataPreprocessor(self.config)

        dh.train_input = data2i_train.transform_input(self.train_input)
        dh.train_output, dh.train_output_scaler, dh.train_output_columns, dh.train_output_dates = data2o_train.transform_output(self.train_output)
        dh.test_input = data2i_test.transform_input(self.test_input)
        dh.test_output, dh.test_output_scaler, dh.test_output_columns, dh.test_output_dates = data2o_test.transform_output(self.test_output)

        if self.context:
            dh.train_context = data2i_train.transform_input(self.train_context)
            dh.test_context = data2i_test.transform_input(self.test_context)
        self.data_holder = dh


def _filter_history_by_dates(df: pd.DataFrame, dates: dict) -> pd.DataFrame:
    if dates is None:
        return df

    from_date = dates['from']
    to_date = dates['to']

    first_available_date = df.iloc[0].date
    last_available_date = df.iloc[-1].date

    if from_date < first_available_date:
        from_index = 0
    else:
        from_index = _get_closest_date_index(df, from_date)

    if to_date > last_available_date:
        to_index = -1
    else:
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
