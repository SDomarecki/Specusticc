from datetime import datetime, timedelta

import pandas as pd

from specusticc.configs_init.model.preprocessor_config import (
    PreprocessorConfig,
    DateRange,
)
from specusticc.data_loading.loaded_data import LoadedData
from specusticc.data_preprocessing.data_set import DataSet
from specusticc.data_preprocessing.input_data_preprocessor import InputDataPreprocessor
from specusticc.data_preprocessing.output_data_preprocessor import (
    OutputDataPreprocessor,
)
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData


class DataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.input: {}
        self.output: {}
        self.context: {}

        self.input_df: pd.DataFrame
        self.output_df: pd.DataFrame
        self.context_df: pd.DataFrame

        self.preprocessed_data: PreprocessedData = PreprocessedData()

    def get_data(self) -> PreprocessedData:
        return self.preprocessed_data

    def preprocess_data(self, data: LoadedData):
        self.input = data.input
        self.output = data.output
        self.context = data.context

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
        unified_df = pd.DataFrame(columns=["date"])
        for ticker, df in data.items():
            df.columns = [
                ticker + "_" + str(col) for col in df.columns if col != "date"
            ] + ["date"]
            unified_df = unified_df.merge(
                df, left_on="date", right_on="date", how="outer"
            )
        unified_df = (
            unified_df.sort_values(by=["date"])
            .interpolate()
            .fillna(1.0)
            .reset_index(drop=True)
        )
        return unified_df

    def _filter_by_dates(self):
        self.train_ioc = {}
        self.train_ioc["input"] = _filter_history_by_dates(
            self.input_df, self.config.train_date
        )
        self.train_ioc["output"] = _filter_history_by_dates(
            self.output_df, self.config.train_date
        )
        self.train_ioc["context"] = _filter_history_by_dates(
            self.context_df, self.config.train_date
        )

        self.test_iocs = []
        for date_range in self.config.test_dates:
            test_ioc = {}
            test_ioc["input"] = _filter_history_by_dates(self.input_df, date_range)
            test_ioc["output"] = _filter_history_by_dates(self.output_df, date_range)
            test_ioc["context"] = _filter_history_by_dates(self.context_df, date_range)
            self.test_iocs.append(test_ioc)

    def _limit_context_dates_by_input_dates(self):
        self.train_ioc["context"] = self.train_ioc["context"][
            self.train_ioc["context"]["date"].isin(self.train_ioc["input"]["date"])
        ]
        for test_ioc in self.test_iocs:
            test_ioc["context"] = test_ioc["context"][
                test_ioc["context"]["date"].isin(test_ioc["input"]["date"])
            ]

    def _reshape_data_to_neural_network(self):
        ph = self.preprocessed_data
        data2i = InputDataPreprocessor(self.config)
        data2o = OutputDataPreprocessor(self.config)

        data_set = DataSet()
        data_set.input = data2i.transform_input(self.train_ioc["input"])
        data_set.context = data2i.transform_input(self.train_ioc["context"])
        (
            data_set.output,
            data_set.output_scaler,
            data_set.output_columns,
            data_set.output_dates,
        ) = data2o.transform_output(self.train_ioc["output"])
        ph.train_set = data_set

        for test_ioc in self.test_iocs:
            data_set = DataSet()
            data_set.input = data2i.transform_input(test_ioc["input"])
            data_set.context = data2i.transform_input(test_ioc["context"])
            (
                data_set.output,
                data_set.output_scaler,
                data_set.output_columns,
                data_set.output_dates,
            ) = data2o.transform_output(test_ioc["output"])
            ph.test_sets.append(data_set)

        self.preprocessed_data = ph


def _filter_history_by_dates(df: pd.DataFrame, dates: DateRange) -> pd.DataFrame:
    if dates is None:
        return df

    from_date = dates.from_date
    to_date = dates.to_date

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
            closest_index = df.index[df["date"] == date][0]
        except Exception:
            pass
        date = date - timedelta(days=1)
    return closest_index
