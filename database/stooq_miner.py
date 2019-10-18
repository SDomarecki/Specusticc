import datetime
import os

import pymongo
import pandas as pd
from models.price_history import PriceHistory
from models.price_record import PriceRecord


class StooqMiner:
    def __init__(self):
        self.now = datetime.datetime.utcnow()
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["stocks"]
        self.tickers = self.fetch_tickers_from_database()

    def fetch_tickers_from_database(self) -> [str]:
        tickers = []
        collection = self.db['companies']
        for company in collection.find():
            tickers.append(company['ticker'])
        return tickers

    def mine(self) -> None:
        phs = []
        i = 0
        print('To fetch: ' + str(len(self.tickers)) + ' companies')
        for ticker in self.tickers:
            prices = self.read_csv(ticker)
            ph = self.dataframe2price_history(ticker, prices)
            phs.append(ph)
            i += 1
            print('Fetched already ' + str(i) + ' companies')
            # self.save_in_database([ph])
        self.save_in_database(phs)

    def read_csv(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.lower()
        df = pd.read_csv(ticker + '_d.csv', delimiter=',')
        return df

    def dataframe2price_history(self, ticker: str, prices: pd.DataFrame) -> PriceHistory:
        ph = PriceHistory(ticker, self.now)
        for index, row in prices.iterrows():
            date_time_str = row['Data']
            date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
            pr = PriceRecord(date_time_obj, row['Otwarcie'],
                             row['Najwyzszy'], row['Najnizszy'],
                             row['Zamkniecie'], row['Wolumen'])
            ph.history.append(pr)
        return ph

    def save_in_database(self, phs: [PriceHistory]) -> None:
        prices = self.db.prices
        dicts = [ph.__dict__() for ph in phs]
        prices.insert_many(dicts)


if __name__ == "__main__":
    os.chdir('stooq_original')

    sm = StooqMiner()
    sm.mine()