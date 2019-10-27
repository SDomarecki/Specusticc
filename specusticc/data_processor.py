import numpy as np
import pandas as pd


def get_closest_date_index(df: pd.DataFrame, date):
    from datetime import timedelta

    closest_index = None
    while closest_index is None:
        try:
            closest_index = df.index[df['date'] == date.strftime('%Y-%m-%d')][0]
        except:
            pass
        date = date - timedelta(days=1)
    return closest_index


def filter_history_by_dates(df: pd.DataFrame, from_date_str: str, to_date_str: str) -> pd.DataFrame:
    from datetime import datetime
    from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
    from_index = get_closest_date_index(df, from_date)

    to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
    to_index = get_closest_date_index(df, to_date)

    return df.iloc[from_index:to_index]


class DataProcessor:
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.ticker = config['data']['ticker']

        dataframe = self.load_from_database()
        i_split = int(len(dataframe) * config['data']['train_test_split'])
        self.data_train = dataframe.get(config['data']['columns']).values[:i_split]
        self.data_test = dataframe.get(config['data']['columns']).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def load_from_database(self) -> pd.DataFrame:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["stocks"]
        prices = db['prices']
        history = prices.find_one({"ticker": self.ticker})['history']
        df = pd.DataFrame(history)[['date', 'open', 'high', 'low', 'close', 'vol']]
        df = filter_history_by_dates(df,
                                          self.config['data']['date']['from'],
                                          self.config['data']['date']['to'])

        return df

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        if normalise:
            data_windows = self.normalise_windows(data_windows, single_window=False)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i + seq_len]
        if normalise:
            window = self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
