import matplotlib.pyplot as plt
import pandas as pd
import os

class SummaryPlotter:
    first_plot = ['mlp', 'gan', 'cnn']
    second_plot = ['lstm', 'encoder-decoder', 'transformer']
    third_plot = ['cnn-lstm', 'lstm-attention']

    first_test_flag = False

    if first_test_flag:
        colors = {
            'true': 'red',
            'mlp': 'dodgerblue',
            'gan': 'green',
            'cnn': 'orange',
            'lstm': 'orangered',
            'encoder-decoder': 'indigo',
            'transformer': 'dodgerblue',
            'cnn-lstm': 'black',
            'lstm-attention': 'brown'
        }
    else:
        colors = {
            'true': 'dodgerblue',
            'lstm-attention': 'red'
        }

    def __init__(self, true_data: pd.DataFrame, preds: dict, config: dict, save_path: str):
        self.true_data = true_data
        self.preds = preds
        self.save_path = save_path + '/plots'
        os.makedirs(self.save_path, exist_ok=True)

        self.rolling = config['preprocessing']['rolling']
        self.horizon = config['preprocessing']['horizon']

    def draw_plots(self):
        self.draw_first_plot()
        self.draw_second_plot()
        self.draw_third_plot()

    def draw_first_plot(self):
        self.draw_one_plot(SummaryPlotter.first_plot, 'first')

    def draw_second_plot(self):
        self.draw_one_plot(SummaryPlotter.second_plot, 'second')

    def draw_third_plot(self):
        self.draw_one_plot(SummaryPlotter.third_plot, 'third')

    def draw_one_plot(self, selected_cols: [str], plot_name: str):
        fig = plt.figure(facecolor='white', figsize=(6.4, 4.8))
        true_date = self.true_data['date']
        target = self.true_data.iloc[:, 0]

        if not self.first_test_flag:
            plt.plot(true_date, target, linewidth=1.5, alpha=1.0, color=self.colors['true'], label='True data')

        first_date = self.true_data.iloc[0, 1]
        last_date = self.true_data.iloc[-1, 1]
        plt.xlim([first_date, last_date])
        plt.xticks(rotation=45)

        max_val = self.true_data.iloc[:, 0].max()
        min_val = self.true_data.iloc[:, 0].min()
        plt.ylim([min_val*0.9, max_val*1.1])

        for name, one_pred in self.preds.items():
            pred_data = one_pred.iloc[:, 0]
            date = one_pred['date']
            for i in range(0, len(date), self.rolling):
                x = date[i:i+self.horizon]
                y = pred_data[i:i+self.horizon]

                if name in selected_cols:
                    if i == 0:
                        plt.plot(x, y, linewidth=1, alpha=0.8, label=name, color=self.colors[name])
                    else:
                        plt.plot(x, y, linewidth=1, alpha=0.8, color=self.colors[name])
                else:
                    plt.plot(x, y, linewidth=0.7, alpha=0.1, color=self.colors[name])

        if self.first_test_flag:
            plt.plot(true_date, target, linewidth=1.5, alpha=1.0, color=self.colors['true'], label='True data')

        plt.grid()
        plt.legend(loc='upper left')
        plt_save_path = self.save_path + '/' + plot_name + '.png'
        fig.tight_layout()

        fig.savefig(plt_save_path)
        fig.show()

