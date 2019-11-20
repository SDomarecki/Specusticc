import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import pandas as pd
from datetime import datetime

class Plotter:
    def __init__(self, config: dict, true_data: pd.DataFrame):
        self.true_data = true_data
        self.config = config
        self.x_len = len(self.true_data)
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
        fig = plt.figure(facecolor='white', dpi=150)
        gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[10, 1, 1])
        self._draw_original_data(fig, gs)
        self._draw_original_classes(fig, gs, original_classes)
        self._draw_predicted_classes(fig, gs, predicted_classes)
        plt.tight_layout()

    def _draw_original_data(self, fig, gs):
        ax = fig.add_subplot(gs[0])
        ax.set_title('Original price chart')

        date_series = self.true_data['date']
        close_series = self.true_data['close']

        first_date = date_series.iloc[0]
        last_date = date_series.iloc[-1]
        self._set_proper_date_xlim(first_date, last_date)
        ax.plot_date(date_series, close_series, color='cornflowerblue', linestyle='solid', markersize=0)
        plt.grid()

    def _draw_original_classes(self, fig, gs, classes):
        plot_title = 'Original price direction'
        ax = self._draw_class_plot(fig, gs[1], plot_title)

        self._draw_classes(classes)

    def _draw_predicted_classes(self, fig, gs, classes):
        plot_title = 'Predicted price direction'
        ax = self._draw_class_plot(fig, gs[2], plot_title)

        prediction_classes = []
        for pred in classes:
            prediction_classes.append(np.argmax(pred))
        self._draw_classes(prediction_classes)
        self._draw_classification_legend(ax)

    def _draw_class_plot(self, fig, spec, title: str):
        ax = fig.add_subplot(spec)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title(title)
        self._set_proper_numeric_xlim()
        return ax

    def _draw_classes(self, classes):
        prediction_shift = self.config['data']['prediction_shift']
        seq_len = self.config['data']['sequence_length']

        for i, data in enumerate(classes):
            padding = i * prediction_shift + seq_len
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
            plt.plot(padding, 1, marker=marker, color=color)

    def _set_proper_numeric_xlim(self):
        axes = plt.gca()
        axes.set_xlim([0, self.x_len])

    def _set_proper_date_xlim(self, from_date: datetime, to_date: datetime):
        axes = plt.gca()
        axes.set_xlim([from_date, to_date])

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
                  fancybox=True, shadow=True, ncol=5)


    def save_prediction_plot(self, path: str):
        save_path = path + '/plot.png'
        plt.savefig(save_path)
