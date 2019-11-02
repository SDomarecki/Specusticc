import pandas as pd
from datetime import datetime, timedelta


def _get_closest_date_index(df: pd.DataFrame, date: datetime) -> int:
    closest_index = None
    while closest_index is None:
        try:
            closest_index = df.index[df['date'] == date.strftime('%Y-%m-%d')][0]
        except Exception:
            pass
        date = date - timedelta(days=1)
    return closest_index


def _filter_history_by_dates(df: pd.DataFrame, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    from_index = _get_closest_date_index(df, from_date)
    to_index = _get_closest_date_index(df, to_date)
    return df.iloc[from_index:to_index]


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
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        stocks = client["stocks"]
        prices = stocks['prices']
        history = prices.find_one({"ticker": ticker})['history']

        df = pd.DataFrame(history)[['date', 'open', 'high', 'low', 'close', 'vol']]

        from_date = datetime.strptime(self.config['import']['date']['from'], '%Y-%m-%d')
        to_date = datetime.strptime(self.config['import']['date']['to'], '%Y-%m-%d')
        df = _filter_history_by_dates(df, from_date, to_date)
        return df
