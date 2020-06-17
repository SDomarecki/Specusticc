from specusticc.configs_init.postprocessor_config import PostprocessorConfig
from specusticc.data_preprocessing.data_holder import DataHolder

import pandas as pd

class DataPostprocessor:
    def __init__(self, processed_data: DataHolder, test_results: dict, config: PostprocessorConfig):
        self.config = config
        self.processed_data = processed_data
        self.test_results = test_results
        self.postprocessed_data = {}
        self.postprocessed_data['test'] = {}
        self.postprocessed_data['learn'] = {}

    def get_data(self):
        self._postprocess()
        return self.postprocessed_data

    def _postprocess(self):
        self._inverse_scales()
        self._retrieve_dataframes()

    def _inverse_scales(self):
        self._inverse_learn_data_scale()
        self._inverse_test_data_scale()

    def _inverse_test_data_scale(self):
        test_scaler = self.processed_data.test_output_scaler
        test_true_output_2D = test_scaler.inverse_transform(self.processed_data.test_output)
        test_predicted_output_2D = test_scaler.inverse_transform(self.test_results['test'])

        self.postprocessed_data['test']['true_data'] = test_true_output_2D.flatten()
        self.postprocessed_data['test']['prediction'] = test_predicted_output_2D.flatten()

    def _inverse_learn_data_scale(self):
        learn_scaler = self.processed_data.train_output_scaler
        learn_true_output_2D = learn_scaler.inverse_transform(self.processed_data.train_output)
        learn_predicted_output_2D = learn_scaler.inverse_transform(self.test_results['learn'])

        self.postprocessed_data['learn']['true_data'] = learn_true_output_2D.flatten()
        self.postprocessed_data['learn']['prediction'] = learn_predicted_output_2D.flatten()

    def _retrieve_dataframes(self):
        self._retrieve_learn_dataframe()
        self._retrieve_test_dataframe()

    def _retrieve_learn_dataframe(self):
        df = pd.DataFrame(data=self.postprocessed_data['learn']['true_data'],
                          index=self.processed_data.train_output_dates.index,
                          columns=self.processed_data.train_output_columns[:-1])
        df['date'] = self.processed_data.train_output_dates
        self.postprocessed_data['learn']['true_data'] = df

        df = pd.DataFrame(data=self.postprocessed_data['learn']['prediction'],
                          index=self.processed_data.train_output_dates.index,
                          columns=self.processed_data.train_output_columns[:-1])
        df['date'] = self.processed_data.train_output_dates
        self.postprocessed_data['learn']['prediction'] = df
        # create dataframe from post...['learn']['true_data'] and processed_data.train_output_columns and .train_output_dates
        # create dataframe from post...['learn']['prediction'] and processed_data.train_output_columns and .train_output_dates
        pass

    def _retrieve_test_dataframe(self):
        df = pd.DataFrame(data=self.postprocessed_data['test']['true_data'],
                          index=self.processed_data.test_output_dates.index,
                          columns=self.processed_data.test_output_columns[:-1])
        df['date'] = self.processed_data.test_output_dates
        self.postprocessed_data['test']['true_data'] = df

        df = pd.DataFrame(data=self.postprocessed_data['test']['prediction'],
                          index=self.processed_data.test_output_dates.index,
                          columns=self.processed_data.test_output_columns[:-1])
        df['date'] = self.processed_data.test_output_dates
        self.postprocessed_data['test']['prediction'] = df
        # create dataframe from post...['test']['true_data'] and processed_data.test_output_columns and .test_output_dates
        # create dataframe from post...['test']['prediction'] and processed_data.test_output_columns and .test_output_dates
        pass