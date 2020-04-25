import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specusticc.configs_init.reporter_config import ReporterConfig
from specusticc.reporting.plotter import Plotter


class RegressionPlotter(Plotter):
    learn_title = 'Learning set price chart'
    test_title = 'Testing set price chart'
    predictions_title = 'Predictions'
    def __init__(self, config: ReporterConfig):
        super().__init__(config)

    def draw_prediction_plot(self, test_results: dict):
        self._draw_regression_plot(test_results)

    def _draw_regression_plot(self, test_results: dict):
        fig, axes = self._draw_empty_figure()
        if self.config.test_on_learning_base:
            self._draw_original_time_series_wo_dates(axes[0][0], self.learn_title, test_results['learn']['true_data'])
            self._draw_predictions(axes[1][0], self.predictions_title, test_results['learn']['prediction'])
            self._draw_original_time_series_wo_dates(axes[0][1], self.test_title, test_results['test']['true_data'])
            self._draw_predictions(axes[1][1], self.predictions_title, test_results['test']['prediction'])
        else:
            self._draw_original_time_series_wo_dates(axes[0], test_results['test']['true_data'])
            self._draw_predictions(axes[1], self.predictions_title, test_results['test']['prediction'])
        self._draw_regression_legend(axes[1][1])
        plt.tight_layout()

    def _draw_empty_figure(self):
        if self.config.test_on_learning_base:
            return plt.subplots(facecolor='white', figsize=(14.4, 9.6), nrows=2, ncols=2,
                                gridspec_kw={'height_ratios': [4, 4]})
        else:
            return plt.subplots(facecolor='white', figsize=(9.6, 7.2), nrows=2,
                                gridspec_kw={'height_ratios': [4, 4]})

    def _draw_predictions(self, ax, title, predictions):
        ax.set_title(title)
        self._set_proper_numeric_xlim(ax, len(predictions))
        x = np.arange(0, len(predictions), 1)
        ax.plot(x, predictions, color='limegreen')
        ax.grid()

    def _draw_regression_legend(self, ax):
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='cornflowerblue', lw=2),
                        Line2D([0], [0], color='limegreen', lw=2)]
        # Put a legend below current axis
        ax.legend(custom_lines, ['True data', 'Prediction'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=7)
