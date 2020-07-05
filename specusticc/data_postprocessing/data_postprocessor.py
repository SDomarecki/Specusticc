from specusticc.data_postprocessing.postprocessed_data import PostprocessedData
from specusticc.data_preprocessing.data_holder import DataHolder

import pandas as pd

from specusticc.model_testing.prediction_results import PredictionResults


class DataPostprocessor:
    def __init__(self, processed_data: DataHolder, test_results: PredictionResults):
        self.processed_data = processed_data
        self.test_results: PredictionResults = test_results
        self.postprocessed_data = PostprocessedData()

    def get_data(self):
        self._postprocess()
        return self.postprocessed_data

    def _postprocess(self):
        self._inverse_scales()
        self._retrieve_dataframes()

    def _inverse_scales(self):
        self._inverse_train_data_scale()
        self._inverse_test_data_scale()

    def _inverse_test_data_scale(self):
        test_scaler = self.processed_data.test_output_scaler
        test_true_output_2D = test_scaler.inverse_transform(self.processed_data.test_output)
        test_predicted_output_2D = test_scaler.inverse_transform(self.test_results.test_output)

        self.postprocessed_data.test_true_data = test_true_output_2D.flatten()
        self.postprocessed_data.test_prediction = test_predicted_output_2D.flatten()

    def _inverse_train_data_scale(self):
        train_scaler = self.processed_data.train_output_scaler
        train_true_output_2D = train_scaler.inverse_transform(self.processed_data.train_output)
        train_predicted_output_2D = train_scaler.inverse_transform(self.test_results.train_output)

        self.postprocessed_data.train_true_data = train_true_output_2D.flatten()
        self.postprocessed_data.train_prediction = train_predicted_output_2D.flatten()

    def _retrieve_dataframes(self):
        self._retrieve_train_dataframe()
        self._retrieve_test_dataframe()

    def _retrieve_train_dataframe(self):
        df = pd.DataFrame(data=self.postprocessed_data.train_true_data,
                          index=self.processed_data.train_output_dates.index,
                          columns=self.processed_data.train_output_columns[:-1])
        df['date'] = self.processed_data.train_output_dates
        self.postprocessed_data.train_true_data = df

        df = pd.DataFrame(data=self.postprocessed_data.train_prediction,
                          index=self.processed_data.train_output_dates.index,
                          columns=self.processed_data.train_output_columns[:-1])
        df['date'] = self.processed_data.train_output_dates
        self.postprocessed_data.train_prediction = df

    def _retrieve_test_dataframe(self):
        df = pd.DataFrame(data=self.postprocessed_data.test_true_data,
                          index=self.processed_data.test_output_dates.index,
                          columns=self.processed_data.test_output_columns[:-1])
        df['date'] = self.processed_data.test_output_dates
        self.postprocessed_data.test_true_data = df

        df = pd.DataFrame(data=self.postprocessed_data.test_prediction,
                          index=self.processed_data.test_output_dates.index,
                          columns=self.processed_data.test_output_columns[:-1])
        df['date'] = self.processed_data.test_output_dates
        self.postprocessed_data.test_prediction = df
