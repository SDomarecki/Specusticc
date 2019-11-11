class PriceRecord:
    def __init__(self
                 , date
                 , open: float
                 , high: float
                 , low: float
                 , close: float
                 , vol: int):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.vol = vol
