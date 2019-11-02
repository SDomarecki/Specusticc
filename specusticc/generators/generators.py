import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def generate(config_import: dict) -> pd.DataFrame:
    from_date = datetime.strptime(config_import['date']['from'], '%Y-%m-%d')
    to_date = datetime.strptime(config_import['date']['to'], '%Y-%m-%d')
    from_date = _skip_weekend(from_date)
    to_date = _skip_weekend(to_date)

    gauss = config_import['gaussian_noise']
    method = config_import['method']

    price_history = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'vol'])
    current_price = 100
    current_date = from_date
    while current_date < to_date:
        if method == 'random_walk':
            step = np.random.choice(a=[-1., 0., 1.])
        elif method == 'sine':
            step = np.sin(price_history.__len__() * np.pi)
        elif method == 'exponental':
            step = current_price*0.001
        elif method == 'polynomial':
            step = 1

        if gauss:
            step += np.random.normal(-step*0.5, step*0.5)

        next_price = current_price + step
        if step > 0:
            high = next_price
            low = current_price
        else:
            high = current_price
            low = next_price

        new_row = pd.Series([current_date, current_price, high, low, next_price, 1000], index=price_history.columns)
        price_history = price_history.append(new_row, ignore_index=True)

        current_price = next_price
        current_date += timedelta(days=1)

        current_date = _skip_weekend(current_date)
    return price_history


def _skip_weekend(date: datetime) -> datetime:
    # Monday - 0
    # Workdays - 0-4
    # Sunday - 6
    if date.weekday() <= 4:
        return date
    return date + timedelta(days=7 - date.weekday())