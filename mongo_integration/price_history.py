from database.price_record import PriceRecord


class PriceHistory:
    def __init__(self, ticker, fetch_date):
        self.ticker = ticker
        self.fetch_date = fetch_date
        self.history: [PriceRecord] = []

    def __dict__(self):
        historyDict = [history.__dict__ for history in self.history]
        return {'ticker': self.ticker, 'fetch_date': self.fetch_date, 'history': historyDict}