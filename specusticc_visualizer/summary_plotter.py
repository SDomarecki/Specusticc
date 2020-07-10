import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

class SummaryPlotter:
    first_plot = ['basic', 'mlp', 'gan']
    second_plot = ['cnn', 'lstm', 'lstmcnn']
    third_plot = ['encoder-decoder', 'lstm-attention', 'transformer_classes']

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.train_data = train_data
        self.test_data = test_data

    def draw_plots(self):
        self.draw_test()
        self.draw_train()

    def draw_train(self):
        self.draw_first_plot(self.train_data)
        self.draw_second_plot(self.train_data)
        self.draw_third_plot(self.train_data)

    def draw_test(self):
        self.draw_first_plot(self.test_data)
        self.draw_second_plot(self.test_data)
        self.draw_third_plot(self.test_data)

    def draw_first_plot(self, data):
        self.draw_one_plot(data, SummaryPlotter.first_plot)

    def draw_second_plot(self, data):
        self.draw_one_plot(data, SummaryPlotter.second_plot)

    def draw_third_plot(self, data):
        self.draw_one_plot(data, SummaryPlotter.third_plot)

    def draw_one_plot(self, data: pd.DataFrame, selected_cols: [str]):
        plt.figure(facecolor='white', figsize=(14.4, 9.6))
        data_columns = data.columns.values.tolist()
        data_columns.remove('date')

        first_date = data.iloc[0]['date']
        last_date = data.iloc[-1]['date']
        self._set_proper_date_xlim(first_date, last_date)
        for col in data_columns:
            if col == data_columns[0]:
                plt.plot(data['date'], data[col], linewidth=1, alpha=1.0, label='True data')
                continue
            name = col.split('_')[2]
            if name in selected_cols:
                plt.plot(data['date'], data[col], linewidth=1, alpha=1.0, label=name)
            else:
                plt.plot(data['date'], data[col], linewidth=0.7, alpha=0.3)
        plt.grid()
        plt.legend()
        plt.show()

    def _set_proper_date_xlim(self, from_date: datetime, to_date: datetime):
        plt.xlim([from_date, to_date])
