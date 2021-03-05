import numpy as np
import pandas as pd

from specusticc.data_postprocessing.postprocessed_data import PostprocessedData
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData
from specusticc.model_testing.prediction_results import PredictionResults


class DataPostprocessor:
    def __init__(
        self, preprocessed_data: PreprocessedData, test_results: PredictionResults
    ):
        self.preprocessed_data = preprocessed_data
        self.test_results: PredictionResults = test_results
        self.postprocessed_data = PostprocessedData()

    def get_data(self) -> PostprocessedData:
        self._postprocess()
        return self.postprocessed_data

    def _postprocess(self):
        self.reverse_train_detrend()
        self.reverse_tests_detrend()

        self._retrieve_train_dataframe()
        self._retrieve_test_dataframes()

    def reverse_train_detrend(self):
        scaler = self.preprocessed_data.train_set.output_scaler
        true_samples = self.preprocessed_data.train_set.output
        predicted_samples = self.test_results.train_output
        reversed_true_samples = np.empty(true_samples.shape)
        reversed_predicted_samples = np.empty(true_samples.shape)
        for i in range(len(true_samples)):
            true_sample = true_samples[i]
            predicted_sample = predicted_samples[i]
            one_scaler = scaler[i]

            reversed_true_sample = np.ones(true_sample.shape)
            reversed_predicted_sample = np.ones(predicted_sample.shape)
            reversed_true_sample[0] = one_scaler
            reversed_predicted_sample[0] = one_scaler
            for j in range(1, len(true_sample)):
                reversed_true_sample[j] = true_sample[j] * one_scaler
                reversed_predicted_sample[j] = predicted_sample[j] * one_scaler

            reversed_true_samples[i] = reversed_true_sample
            reversed_predicted_samples[i] = reversed_predicted_sample
        self.postprocessed_data.train_true_data = reversed_true_samples.flatten()
        self.postprocessed_data.train_prediction = reversed_predicted_samples.flatten()

    def reverse_tests_detrend(self):
        for i in range(len(self.preprocessed_data.test_sets)):
            scaler = self.preprocessed_data.test_sets[i].output_scaler
            true_samples = self.preprocessed_data.test_sets[i].output
            predicted_samples = self.test_results.test_output[i]
            reversed_true_samples = np.empty(true_samples.shape)
            reversed_predicted_samples = np.empty(true_samples.shape)
            for j in range(len(true_samples)):
                true_sample = true_samples[j]
                predicted_sample = predicted_samples[j]
                one_scaler = scaler[j]

                reversed_true_sample = np.ones(true_sample.shape)
                reversed_predicted_sample = np.ones(predicted_sample.shape)
                reversed_true_sample[0] = one_scaler
                reversed_predicted_sample[0] = one_scaler
                for k in range(1, len(true_sample)):
                    reversed_true_sample[k] = true_sample[k] * one_scaler
                    reversed_predicted_sample[k] = predicted_sample[k] * one_scaler

                reversed_true_samples[j] = reversed_true_sample
                reversed_predicted_samples[j] = reversed_predicted_sample
            self.postprocessed_data.test_true_datas.append(
                reversed_true_samples.flatten()
            )
            self.postprocessed_data.test_predictions.append(
                reversed_predicted_samples.flatten()
            )

    def _retrieve_train_dataframe(self):
        df = pd.DataFrame(
            data=self.postprocessed_data.train_true_data,
            index=self.preprocessed_data.train_set.output_dates.index,
            columns=self.preprocessed_data.train_set.output_columns[:-1],
        )
        df["date"] = self.preprocessed_data.train_set.output_dates
        self.postprocessed_data.train_true_data = df

        df = pd.DataFrame(
            data=self.postprocessed_data.train_prediction,
            index=self.preprocessed_data.train_set.output_dates.index,
            columns=self.preprocessed_data.train_set.output_columns[:-1],
        )
        df["date"] = self.preprocessed_data.train_set.output_dates
        self.postprocessed_data.train_prediction = df

    def _retrieve_test_dataframes(self):
        for i in range(len(self.postprocessed_data.test_true_datas)):
            df = pd.DataFrame(
                data=self.postprocessed_data.test_true_datas[i],
                index=self.preprocessed_data.test_sets[i].output_dates.index,
                columns=self.preprocessed_data.test_sets[i].output_columns[:-1],
            )
            df["date"] = self.preprocessed_data.test_sets[i].output_dates
            self.postprocessed_data.test_true_datas[i] = df

        for i in range(len(self.postprocessed_data.test_predictions)):
            df = pd.DataFrame(
                data=self.postprocessed_data.test_predictions[i],
                index=self.preprocessed_data.test_sets[i].output_dates.index,
                columns=self.preprocessed_data.test_sets[i].output_columns[:-1],
            )
            df["date"] = self.preprocessed_data.test_sets[i].output_dates
            self.postprocessed_data.test_predictions[i] = df
