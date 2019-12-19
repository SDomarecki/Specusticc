import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specusticc.agent.reporting.plotter import Plotter


class RegressionPlotter(Plotter):
    def __init__(self, config: dict, true_data: pd.DataFrame):
        super().__init__(config, true_data)

    def draw_and_save_prediction_plots(self, output_test: dict, predictions: dict, report_directory: str):
        for k, v in self.true_data.items():
            self._draw_regression_plot(predictions, output_test)
            self._save_prediction_plot(report_directory, k)

    def _draw_regression_plot(self, predictions, original_time_series):
        fig, axes = self._draw_empty_figure()
        self._draw_original_time_series_wo_dates(axes[0], original_time_series)
        self._draw_predictions(axes[1], predictions)
        self._draw_regression_legend(axes[1])
        plt.tight_layout()

    def _draw_predictions(self, ax, predictions):
        ax.set_title('Predictions')

        sample_time_difference = self.config['preprocessing']['sample_time_difference']
        seq_len = self.config['preprocessing']['sequence_length']
        prediction_len = self.config['preprocessing']['sequence_prediction_time']

        for i, data in enumerate(predictions):
            start = i * sample_time_difference + seq_len
            end = start + prediction_len
            padding = np.arange(start, end).tolist()
            ax.plot(padding, data, color='limegreen')

        # self._set_proper_numeric_xlim(ax)
        ax.grid()

    def _draw_empty_figure(self):
        return plt.subplots(facecolor='white', figsize=(9.6, 7.2), nrows=2,
                            gridspec_kw={'height_ratios': [4, 4]})

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
