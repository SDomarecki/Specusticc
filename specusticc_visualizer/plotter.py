from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specusticc.configs_init.reporter_config import ReporterConfig


class Plotter:
    def __init__(self, config: ReporterConfig):
        self.config = config
        pd.plotting.register_matplotlib_converters()

    def _set_proper_numeric_xlim(self, ax, x_lim):
        ax.set_xlim([0, x_lim])

    def _set_proper_date_xlim(self, ax, from_date: datetime, to_date: datetime):
        ax.set_xlim([from_date, to_date])

    def save_prediction_plot(self, ticker: str):
        save_path = self.config.report_directory + '/' + ticker + '.png'
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

    def _draw_original_time_series_wo_dates(self, ax, title, time_series):
        ax.set_title(title)
        self._set_proper_numeric_xlim(ax, len(time_series))
        x = np.arange(0, len(time_series), 1)
        ax.plot(x, time_series, color='cornflowerblue')
        ax.grid()

