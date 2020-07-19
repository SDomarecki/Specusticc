import matplotlib.pyplot as plt
import pandas as pd


class SummaryPlotter:
    first_plot = ['basic', 'mlp', 'gan']
    second_plot = ['cnn', 'lstm', 'lstmcnn']
    third_plot = ['encoder-decoder', 'lstm-attention', 'transformer']
    colors = {
        'basic': 'red',
        'mlp': 'orange',
        'gan': 'green',
        'cnn': 'lime',
        'lstm': 'orangered',
        'lstmcnn': 'black',
        'encoder-decoder': 'indigo',
        'lstm-attention': 'brown',
        'transformer': 'dodgerblue'
    }

    def __init__(self, true_data: pd.DataFrame, preds: dict, config: dict):
        self.true_data = true_data
        self.preds = preds

        self.sample_time_difference = config['preprocessing']['sample_time_difference']
        self.sequence_prediction_time = config['preprocessing']['sequence_prediction_time']

    def draw_plots(self):
        self.draw_first_plot()
        self.draw_second_plot()
        self.draw_third_plot()

    def draw_first_plot(self):
        self.draw_one_plot(SummaryPlotter.first_plot)

    def draw_second_plot(self):
        self.draw_one_plot(SummaryPlotter.second_plot)

    def draw_third_plot(self):
        self.draw_one_plot(SummaryPlotter.third_plot)

    def draw_one_plot(self, selected_cols: [str]):
        plt.figure(facecolor='white', figsize=(14.4, 9.6))
        true_date = self.true_data['date']
        target = self.true_data.iloc[:, 0]
        plt.plot(true_date, target, linewidth=1, alpha=1.0, label='True data')

        first_date = self.true_data.iloc[0, 1]
        last_date = self.true_data.iloc[-1, 1]
        plt.xlim([first_date, last_date])

        for name, one_pred in self.preds.items():
            pred_data = one_pred.iloc[:, 0]
            date = one_pred['date']
            for i in range(0, len(date), self.sample_time_difference):
                x = date[i:i+self.sequence_prediction_time]
                y = pred_data[i:i+self.sequence_prediction_time]

                if name in selected_cols:
                    if i == 0:
                        plt.plot(x, y, linewidth=1, alpha=1.0, label=name, color=self.colors[name])
                    else:
                        plt.plot(x, y, linewidth=1, alpha=1.0, color=self.colors[name])
                else:
                    plt.plot(x, y, linewidth=0.7, alpha=0.3, color=self.colors[name])

        plt.grid()
        plt.legend()
        plt.show()