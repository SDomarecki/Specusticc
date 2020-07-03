from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


class Generator:
    @staticmethod
    def generate_test_data():
        fs = 10000  # sample rate
        f = 25  # the frequency of the signal

        x = np.arange(fs)  # the points on the x axis for plotting
        # compute the value (amplitude) of the sin wave at the for each sample
        y = np.sin(2 * np.pi * f * (x / fs)) * 50 + 100
        plt.plot(x, y)
        plt.show()

        current_date = datetime(2000, 1, 1)
        td = timedelta(days=1)
        res = []
        for val in y:
            one_dist = {'date': current_date, 'open': val, 'high': val, 'low': val, 'close': val, 'vol': 100000}
            res.append(one_dist)
            current_date = current_date + td
        return res
