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
        self._save_one_data_csv(self._test_results.test_true_data, 'true_test_data.csv')
        self._save_one_data_csv(self._test_results.test_prediction, 'prediction_test_data.csv')

    def _save_one_data_csv(self, data: pd.DataFrame, filename: str):
        full_path = f'{self._save_path}/{filename}'
        data.to_csv(full_path, index=False, header=True)
