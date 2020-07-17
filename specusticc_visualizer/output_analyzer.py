import json

from specusticc_visualizer.summary_plotter import SummaryPlotter
from os import listdir
import pandas as pd

from specusticc_visualizer.tabular_error import TabularError


class OutputAnalyzer:
    def __init__(self, output_path: str):
        self.output_path = output_path
        config_path = self.output_path + '/config.json'
        self.config = self._load_config(config_path)

    def run(self):
        self.load_all_results()
        self.compile_tabular_summary()
        self.draw_summary_plot()

    def load_all_results(self):
        dirs = listdir(self.output_path)
        dirs.remove('config.json')

        self.load_true_train(dirs)
        self.load_pred_train(dirs)
        self.load_true_test(dirs)
        self.load_pred_test(dirs)

    def load_true_train(self, dirs):
        first_dir = dirs[0]
        true_train_path = f'{self.output_path}/{first_dir}/true_train_data.csv'
        true_train = pd.read_csv(true_train_path)
        true_train['date'] = pd.to_datetime(true_train['date'], format='%Y-%m-%d')
        self.true_train = true_train

    def load_pred_train(self, dirs):
        pred_train_name_groupped = {}
        for dir in dirs:
            pred_train_path = f'{self.output_path}/{dir}/prediction_train_data.csv'
            pred_train = pd.read_csv(pred_train_path)
            pred_train['date'] = pd.to_datetime(pred_train['date'], format='%Y-%m-%d')
            pred_train.columns = [col + '_' + dir for col in pred_train.columns if col != 'date'] + ['date']

            split_dir = dir.split('_')
            name = split_dir[0]
            nr = split_dir[1]
            if nr == '1':
                pred_train_name_groupped[name] = pred_train
            else:
                pred_train_name_groupped[name] = pred_train_name_groupped[name].merge(pred_train, left_on='date', right_on='date', how='outer')
        self.pred_train = pred_train_name_groupped


    def load_true_test(self, dirs):
        first_dir = dirs[0]
        true_test_path = f'{self.output_path}/{first_dir}/true_test_data.csv'
        true_test = pd.read_csv(true_test_path)
        true_test['date'] = pd.to_datetime(true_test['date'], format='%Y-%m-%d')
        self.true_test = true_test

    def load_pred_test(self, dirs):
        pred_test_name_groupped = {}

        for dir in dirs:
            pred_test_path = f'{self.output_path}/{dir}/prediction_test_data.csv'
            pred_test = pd.read_csv(pred_test_path)
            pred_test['date'] = pd.to_datetime(pred_test['date'], format='%Y-%m-%d')
            pred_test.columns = [col + '_' + dir for col in pred_test.columns if col != 'date'] + ['date']

            split_dir = dir.split('_')
            name = split_dir[0]
            nr = split_dir[1]
            if nr == '1':
                pred_test_name_groupped[name] = pred_test
            else:
                pred_test_name_groupped[name] = pred_test_name_groupped[name].merge(pred_test, left_on='date',
                                                                                    right_on='date', how='outer')
        self.pred_test = pred_test_name_groupped


    def compile_tabular_summary(self):
        tabular_error = TabularError()
        print('============================')
        print('Training data errors')
        tabular_error.count_errors(self.true_train, self.pred_train)
        print('============================')
        print('Testing data errors')
        tabular_error.count_errors(self.true_test, self.pred_test)

    def draw_summary_plot(self):
        plotter_train_data = self.true_train
        for name_groupped_dfs in self.pred_train.values():
            df = name_groupped_dfs.iloc[:, 0:2]
            plotter_train_data = plotter_train_data.merge(df, left_on='date', right_on='date', how='outer')

        plotter_test_data = self.true_test
        for name_groupped_dfs in self.pred_test.values():
            df = name_groupped_dfs.iloc[:, 0:2]
            plotter_test_data = plotter_test_data.merge(df, left_on='date', right_on='date', how='outer')

        sp = SummaryPlotter(plotter_train_data, plotter_test_data, self.config)
        sp.draw_plots()

    def _load_config(self, config_path: str) -> dict:
        file = open(config_path)
        return json.load(file)