import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from specusticc.agent import Agent


class Reporter:
    def __init__(self, agent: Agent) -> None:
        self.config = agent.config
        self.input_test, self.output_test = agent.org_test, agent.output_test
        self.predictions = agent.predictions
        self.model = agent.model
        self.report_directory = None

    def print_report(self):
        self._create_report_directory()
        self._save_model()
        self._save_config()
        self._save_prediction_plot()

    def _create_report_directory(self) -> None:
        import specusticc.directories as dirs
        self.report_directory = 'output/' + dirs.get_timestamp_dir()
        dirs.create_save_dir(self.report_directory)

    def _save_model(self):
        save_path = self.report_directory + "/model.h5"
        self.model.save(save_path)

    def _save_config(self):
        import json
        save_path = self.report_directory + '/used_config.json'
        with open(save_path, 'w') as outfile:
            json.dump(self.config, outfile, indent=4)

    def _save_prediction_plot(self):
        target = self.config['model']['target']
        if target == 'regression':
            self._save_regression_report()
        elif target == 'classification':
            self._save_classification_report()
        else:
            raise NotImplementedError

    def _save_regression_report(self):
        seq_len = self.config['data']['sequence_length']

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(self.output_test, label='True Data')
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(self.predictions):
            padding = [None for p in range(i * seq_len)]
            plt.plot(padding + data, label='Prediction')

        plt.grid()

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)

        self._save_plot()

    def _save_classification_report(self):
        fig = plt.figure(facecolor='white', dpi=150)
        gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[10, 1, 1])
        self._draw_original_data(fig, gs)
        self._draw_original_classes(fig, gs)
        self._draw_predicted_classes(fig, gs)
        plt.tight_layout()
        self._save_plot()

    def _draw_original_data(self, fig, gs):
        ax = fig.add_subplot(gs[0])
        ax.set_title('Original price chart')
        self._set_proper_xlim()

        y = self.input_test[['close']].to_numpy()
        ax.plot(y, color='cornflowerblue')
        plt.grid()

    def _draw_original_classes(self, fig, gs):
        plot_title = 'Original price direction'
        ax = self._draw_class_plot(fig, gs[1], plot_title)

        self._draw_classes(self.output_test)

    def _draw_predicted_classes(self, fig, gs):
        plot_title = 'Predicted price direction'
        ax = self._draw_class_plot(fig, gs[2], plot_title)

        self._draw_classes(self.predictions)
        self._create_plot_legend(ax)

    def _draw_class_plot(self, fig, spec, title: str):
        ax = fig.add_subplot(spec)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title(title)
        self._set_proper_xlim()
        return ax

    def _draw_classes(self, classes):
        model_type = self.config['model']['type']
        prediction_shift = self.config['data']['prediction_shift']
        seq_len = self.config['data']['sequence_length']

        for i, data in enumerate(classes):
            padding = i * prediction_shift + seq_len
            if model_type == 'decision_tree':
                arg_max = _label_to_arg_max(data)
            else:
                arg_max = np.argmax(data)
            if arg_max > 2:
                marker = '^'
                color = 'limegreen'
            elif arg_max == 2:
                marker = '>'
                color = 'gold'
            else:
                marker = 'v'
                color = 'crimson'
            plt.plot(padding, 1, marker=marker, color=color)

    def _set_proper_xlim(self):
        axes = plt.gca()
        axes.set_xlim([0, len(self.input_test)])

    def _create_plot_legend(self, ax):
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


    def _save_plot(self):
        save_path = self.report_directory + '/plot.png'
        plt.savefig(save_path)


def _label_to_arg_max(data: str) -> int:
    if data == 'Strong Buy':
        return 4
    elif data == 'Buy':
        return 3
    elif data == 'Hold':
        return 2
    elif data == 'Sell':
        return 1
    else:
        return 0