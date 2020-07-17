import specusticc.utilities.directories as dirs
from specusticc.data_postprocessing.postprocessed_data import PostprocessedData
import pandas as pd

class Reporter:
    def __init__(self, test_results: PostprocessedData, save_path: str):
        self._test_results = test_results
        self._save_path = save_path

    def save_results(self):
        dirs.create_save_dir(self._save_path)

        self._save_one_data_csv(self._test_results.train_true_data, 'true_train_data.csv')
        self._save_one_data_csv(self._test_results.train_prediction, 'prediction_train_data.csv')

        for i in range(len(self._test_results.test_true_datas)):
            true_data = self._test_results.test_true_datas[i]
            filename = f'true_test_data_{i}.csv'
            self._save_one_data_csv(true_data, filename)

        for i in range(len(self._test_results.test_predictions)):
            prediction = self._test_results.test_predictions[i]
            filename = f'prediction_test_data_{i}.csv'
            self._save_one_data_csv(prediction, filename)

    def _save_one_data_csv(self, data: pd.DataFrame, filename: str):
        full_path = f'{self._save_path}/{filename}'
        data.to_csv(full_path, index=False, header=True)
