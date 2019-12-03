import pandas as pd
from datetime import datetime, timedelta
import pymongo


class DataLoader:
    database_url = 'mongodb://localhost:27017/'

    def __init__(self, config: dict) -> None:
        self.config = config
        self.collection = None
        self.train_input = {}
        self.train_output = {}
        self.test_input = {}
        self.test_output = {}

        self._set_collection()

    def load_data(self) -> tuple:
        self._load_dataframes()
        return self.train_input, self.train_output, self.test_input, self.test_output

    def _set_collection(self):
        client = pymongo.MongoClient(DataLoader.database_url)
        stocks_database = client['stocks']
        self.collection = stocks_database['prices']

    def _load_dataframes(self) -> None:
        train_date = self.config['import']['train_date']
        test_date = self.config['import']['test_date']

        for input in self.config['import']['input']:
            ticker = input['ticker']
            loaded = self._load_dataframe(input)
            print('[Loader] Filtering by date')
            one_train_input = _filter_history_by_dates(loaded, train_date)
            self.train_input[ticker] = one_train_input
            one_test_input = _filter_history_by_dates(loaded, test_date)
            print('[Loader] Filtered')
            self.test_input[ticker] = one_test_input

        for target in self.config['import']['target']:
            ticker = target['ticker']
            loaded = self._load_dataframe(target)
            one_train_output = _filter_history_by_dates(loaded, train_date)
            self.train_output[ticker] = one_train_output
            one_test_output = _filter_history_by_dates(loaded, test_date)
            self.test_output[ticker] = one_test_output

    def _load_dataframe(self, input: dict) -> pd.DataFrame:
        print('[Loader] Loading %s' % input)
        ticker = input['ticker']
        # TODO generation case
        raw_history = self.collection.find_one({"ticker": ticker})['history']

        columns = input['columns']
        if 'date' not in columns:
            columns.append('date')

        df_full_history = pd.DataFrame(raw_history)[columns]
        df_full_history['date'] = pd.to_datetime(df_full_history['date'], format='%Y-%m-%d')
        print('[Loader] Loaded')
        return df_full_history


def _filter_history_by_dates(df: pd.DataFrame, dates: dict) -> pd.DataFrame:
    from_date = dates['from']
    to_date = dates['to']
    from_index = _get_closest_date_index(df, from_date)
    to_index = _get_closest_date_index(df, to_date)
    filtered = df.iloc[from_index:to_index]
    return filtered


def _get_closest_date_index(df: pd.DataFrame, date: datetime) -> int:
    closest_index = None
    while closest_index is None:
        try:
            closest_index = df.index[df['date'] == date][0]
        except Exception:
            pass
        date = date - timedelta(days=1)
    return closest_index
