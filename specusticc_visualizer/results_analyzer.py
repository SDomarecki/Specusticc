import json

from specusticc_visualizer.summary_plotter import SummaryPlotter
from os import listdir
import pandas as pd

from specusticc_visualizer.tabular_error import TabularError


class ResultsAnalyzer:
    def __init__(self, output_path: str):
        self.output_path = output_path
        config_path = self.output_path + "/config.json"
        self.config = self._load_config(config_path)

    def analyze_predictions(self):
        dirs = listdir(self.output_path)
        dirs.remove("config.json")

        for one_dir in dirs:
            self.analyze_one_prediction(one_dir)

    def analyze_one_prediction(self, directory):
        full_path = self.output_path + "/" + directory
        files = listdir(full_path)
        if "plots" in files:
            files.remove("plots")
        self.load_all_files(files, full_path)

    def load_all_files(self, files: [], path_to_files: str):
        true_data = None
        preds_groupped = {}
        preds_to_plot = {}
        for filename in files:
            full_path = f"{path_to_files}/{filename}"
            csv = pd.read_csv(full_path)
            csv["date"] = pd.to_datetime(csv["date"], format="%Y-%m-%d")
            if filename == "true_data.csv":
                true_data = csv
                continue

            split_filename = filename.split("_")
            name = split_filename[0]
            nr = split_filename[1].replace(".csv", "")
            if nr == "1":
                preds_groupped[name] = csv
                preds_to_plot[name] = csv
            else:
                csv.columns = [
                    col + "_" + str(nr) for col in csv.columns if col != "date"
                ] + ["date"]
                preds_groupped[name] = preds_groupped[name].merge(
                    csv, left_on="date", right_on="date", how="outer"
                )

        tabular_error = TabularError()
        print("============================")
        print(f"Errors for dir: {path_to_files}")
        tabular_error.count_errors(true_data, preds_groupped)

        sp = SummaryPlotter(
            true_data, preds_to_plot, self.config, save_path=path_to_files
        )
        sp.draw_plots()

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path) as file:
            return json.load(file)
