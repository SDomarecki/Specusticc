import os

import pandas as pd

from specusticc.data_postprocessing.postprocessed_data import PostprocessedData


class Reporter:
    def __init__(
        self, test_results: PostprocessedData, save_path: str, model_name: str
    ):
        self._test_results = test_results
        self._save_path = save_path
        self._model_name = model_name

    def save_results(self):
        os.makedirs(self._save_path, exist_ok=True)

        self._save_one_data_csv(
            self._test_results.train_true_data, "train/true_data.csv"
        )
        self._save_one_data_csv(
            self._test_results.train_prediction, f"train/{self._model_name}.csv"
        )

        for i in range(len(self._test_results.test_true_datas)):
            true_data = self._test_results.test_true_datas[i]
            filename = f"test_{i}/true_data.csv"
            self._save_one_data_csv(true_data, filename)

        for i in range(len(self._test_results.test_predictions)):
            prediction = self._test_results.test_predictions[i]
            filename = f"test_{i}/{self._model_name}.csv"
            self._save_one_data_csv(prediction, filename)

    def _save_one_data_csv(self, data: pd.DataFrame, filename: str):
        full_path = f"{self._save_path}/{filename}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        data.to_csv(full_path, index=False, header=True)
