class PriceHistory:
    def __init__(self, ticker, fetch_date):
        self.ticker = ticker
        self.fetch_date = fetch_date
        self.history = []  # array of PriceRecord!
