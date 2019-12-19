from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotter:
    def __init__(self, config: dict, true_data: dict):
        self.true_data = true_data
        self.config = config
        self.x_len = 0
        pd.plotting.register_matplotlib_converters()

    def draw_and_save_prediction_plots(self, output_test, predictions, report_directory):
        raise NotImplementedError

    def _set_proper_numeric_xlim(self, ax):
        ax.set_xlim([0, self.x_len])

    def _set_proper_date_xlim(self, ax, from_date: datetime, to_date: datetime):
        ax.set_xlim([from_date, to_date])

    def _save_prediction_plot(self, path: str, ticker: str):
        save_path = path + '/' + ticker + '.png'
        plt.savefig(save_path)

    def _draw_original_time_series(self, ax, time_series):
        ax.set_title('Original price chart')

        date_series = time_series['date']
        close_series = time_series['close']
        self.x_len = len(date_series)

        first_date = date_series.iloc[0]
        last_date = date_series.iloc[-1]
        self._set_proper_date_xlim(ax, first_date, last_date)
        ax.plot_date(date_series, close_series, color='cornflowerblue', linestyle='solid', markersize=0)
        ax.grid()

    def _draw_original_time_series_wo_dates(self, ax, time_series):

        sample_time_difference = self.config['preprocessing']['sample_time_difference']
        seq_len = self.config['preprocessing']['sequence_length']
        prediction_len = self.config['preprocessing']['sequence_prediction_time']

        ax.set_title('Original price chart')

        self.x_len = len(time_series)

        for i, data in enumerate(time_series):
            start = i * sample_time_difference + seq_len
            end = start + prediction_len
            padding = np.arange(start, end).tolist()
            ax.plot(padding, data, color='cornflowerblue')

        ax.grid()

