import matplotlib.pyplot as plt
import numpy as np

from specusticc.configs_init.model.reporter_config import ReporterConfig
from specusticc_visualizer.plotter import Plotter


class ClassificationPlotter(Plotter):
    def __init__(self, config: ReporterConfig):
        super().__init__(config)

    def draw_and_save_prediction_plots(self, true_classes: dict, predicted_classes: dict, report_directory: str):
        for k, v in self.true_data.items():
            self._draw_classification_plot(v, true_classes[k], predicted_classes[k])
            self._save_prediction_plot(report_directory, k)

    def _draw_classification_plot(self, original_time_series, original_classes, predicted_classes):
        fig, axes = self._draw_empty_figure()
        self._draw_original_time_series(axes[0], original_time_series)
        self._draw_classes_plot(axes[1], original_classes, 'Original price direction')
        self._draw_classes_plot(axes[2], predicted_classes, 'Predicted price direction')
        self._draw_classification_legend(axes[2])
        plt.tight_layout()

    def _draw_empty_figure(self):
        return plt.subplots(facecolor='white', figsize=(9.6, 7.2), nrows=3,
                            gridspec_kw={'height_ratios': [10, 1, 1]})

    def _draw_classes_plot(self, ax, class_probabilities, plot_title):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title(plot_title)
        self._set_proper_numeric_xlim(ax)

        classes = []
        for class_prob in class_probabilities:
            classes.append(np.argmax(class_prob))
        self._draw_classes(ax, classes)


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
