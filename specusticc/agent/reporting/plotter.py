import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import pandas as pd
from datetime import datetime


class Plotter:
    def __init__(self, config: dict, true_data: pd.DataFrame):
        self.true_data = true_data
        self.config = config
        self.x_len = 0
        pd.plotting.register_matplotlib_converters()

    def draw_regression_plot(self, predictions):
        seq_len = self.config['data']['sequence_length']

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(self.true_data, label='True Data')
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predictions):
            padding = [None for p in range(i * seq_len)]
            plt.plot(padding + data, label='Prediction')

        plt.grid()

        self._draw_regression_legend(ax)

    def _draw_regression_legend(self, ax):
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)

    def draw_classification_plot(self, original_classes, predicted_classes):
        fig, axes = plt.subplots(facecolor='white', figsize=(9.6, 7.2), nrows=3,
                                 gridspec_kw={'height_ratios': [10, 1, 1]})
        self._draw_original_data(axes[0])
        self._draw_original_classes(axes[1], original_classes)
        self._draw_predicted_classes(axes[2], predicted_classes)
        plt.tight_layout()

    def _draw_original_data(self, ax):
        ax.set_title('Original price chart')

        # TODO unfuck this hardcode asap
        date_series = self.true_data['PKO']['date']
        close_series = self.true_data['PKO']['close']
        self.x_len = len(date_series)

        first_date = date_series.iloc[0]
        last_date = date_series.iloc[-1]
        self._set_proper_date_xlim(ax, first_date, last_date)
        ax.plot_date(date_series, close_series, color='cornflowerblue', linestyle='solid', markersize=0)
        ax.grid()

    def _draw_original_classes(self, ax, classes):
        plot_title = 'Original price direction'
        ax = self._draw_class_plot(ax, plot_title)

        self._draw_classes(ax, classes)

    def _draw_predicted_classes(self, ax, classes):
        plot_title = 'Predicted price direction'
        ax = self._draw_class_plot(ax, plot_title)

        prediction_classes = []
        for pred in classes:
            prediction_classes.append(np.argmax(pred))
        self._draw_classes(ax, prediction_classes)
        self._draw_classification_legend(ax)

    def _draw_class_plot(self, ax, title: str):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title(title)
        self._set_proper_numeric_xlim(ax)
        return ax

    def _draw_classes(self, ax, classes):
        sample_time_difference = self.config['preprocessing']['sample_time_difference']
        seq_len = self.config['preprocessing']['sequence_length']

        for i, data in enumerate(classes):
            padding = i * sample_time_difference + seq_len
            if data == 4:
                marker = '^'
                color = 'green'
            elif data == 3:
                marker = '^'
                color = 'limegreen'
            elif data == 2:
                marker = '>'
                color = 'gold'
            elif data == 1:
                marker = 'v'
                color = 'crimson'
            else:
                marker = 'v'
                color = 'red'
            ax.plot(padding, 1, marker=marker, color=color, markersize=10)

    def _set_proper_numeric_xlim(self, ax):
        ax.set_xlim([0, self.x_len])

    def _set_proper_date_xlim(self, ax, from_date: datetime, to_date: datetime):
        ax.set_xlim([from_date, to_date])

    def _draw_classification_legend(self, ax):
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='cornflowerblue', lw=2),
                        Line2D([0], [0], color='w', marker='^', markersize=10, markerfacecolor='limegreen'),
                        Line2D([0], [0], color='w', marker='>', markersize=10, markerfacecolor='gold'),
                        Line2D([0], [0], color='w', marker='v', markersize=10, markerfacecolor='crimson')]
        # Put a legend below current axis
        ax.legend(custom_lines, ['True Data', 'Buy', 'Hold', 'Sell'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=7)

    def save_prediction_plot(self, path: str):
        save_path = path + '/plot.png'
        plt.savefig(save_path)
