from specusticc.agent import Agent
import matplotlib.pyplot as plt
import numpy as np

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


        save_path = self.report_directory + '/plot.png'
        plt.savefig(save_path)

    def _save_classification_report(self):
        prediction_shift = self.config['data']['prediction_shift']
        model_type = self.config['model']['type']
        y = self.input_test[['close']].to_numpy()
        fig = plt.figure(facecolor='white', dpi=150)
        ax = fig.add_subplot(111)
        ax.plot(y, color='cornflowerblue', label='True Data')
        for i, data in enumerate(self.predictions):
            padding = (i+1) * prediction_shift
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

        plt.grid()

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