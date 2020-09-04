import datetime
import os

import pymongo
import pandas as pd
from mongo_integration.price_history import PriceHistory
from mongo_integration.price_record import PriceRecord


class StooqMiner:
    def __init__(self):
        self.now = datetime.datetime.utcnow()
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["stocks"]
        self.files = os.listdir()

    def mine(self):
        price_histories = []
        i = 0
        print(f'To fetch: {len(self.files)} companies')
        for file in self.files:
            ticker = file.replace('_d.csv', '')
            prices = self.read_csv(file)
            price_history = self.dataframe2price_history(ticker, prices)
            price_histories.append(price_history)
            i += 1
            print(f'Fetched already {i} companies', end='\r')
        print('Saving in database...', end='')
        self.save_in_database(price_histories)
        print('Saved')

    def read_csv(self, file: str) -> pd.DataFrame:
        file = file.lower()
        df = pd.read_csv(file, delimiter=',')
        return df

    def dataframe2price_history(self, ticker: str, prices: pd.DataFrame) -> PriceHistory:
        ph = PriceHistory(ticker, self.now)
        for index, row in prices.iterrows():
            date_time_str = row['Data']
            date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
            pr = PriceRecord(date_time_obj, row['Otwarcie'],
                             row['Najwyzszy'], row['Najnizszy'],
                             row['Zamkniecie'])
            if 'Wolumen' in row:
                pr.vol = row['Wolumen']
            ph.history.append(pr)
        return ph

    def save_in_database(self, phs: [PriceHistory]):
        prices = self.db.prices
        dicts = [ph.__dict__() for ph in phs]
        prices.delete_many({}) #Delete all records in collection
        prices.insert_many(dicts)


if __name__ == "__main__":
    os.chdir('stooq_original')

    sm = StooqMiner()
    sm.mine()