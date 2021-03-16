class PriceRecord:
    def __init__(
        self,
        date,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ):
        self.date = date
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
