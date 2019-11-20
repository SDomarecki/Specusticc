import pandas as pd
from datetime import datetime, timedelta
import pymongo


def _get_closest_date_index(df: pd.DataFrame, date: datetime) -> int:
    closest_index = None
    while closest_index is None:
        try:
            closest_index = df.index[df['date'] == date][0]
        except Exception:
            pass
        date = date - timedelta(days=1)
    return closest_index


def _filter_history_by_dates(df: pd.DataFrame, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    from_index = _get_closest_date_index(df, from_date)
    to_index = _get_closest_date_index(df, to_date)
    filtered = df.iloc[from_index:to_index]
    return filtered


class DataLoader:
    def __init__(self, config: dict) -> None:
        self.config = config

    def load_data(self) -> pd.DataFrame:
        datatype = self.config['import']['datatype']
        if datatype == 'generated':
            from specusticc.generators.generators import generate
            return generate(self.config['import'])
        elif datatype == 'real':
            return self._load_from_database()
        else:
            raise NotImplementedError

    def _load_from_database(self) -> pd.DataFrame:
        ticker = self.config['import']['ticker']
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        stocks = client["stocks"]
        prices = stocks['prices']
        raw_history = prices.find_one({"ticker": ticker})['history']

        columns = self.config['data']['columns']
        df_full_history = pd.DataFrame(raw_history)[columns]
        df_full_history['date'] = pd.to_datetime(df_full_history['date'], format='%Y-%m-%d')

        from_date = self.config['import']['date']['from']
        to_date = self.config['import']['date']['to']
        df_filtered_history = _filter_history_by_dates(df_full_history, from_date, to_date)
        return df_filtered_history
