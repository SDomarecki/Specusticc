class StatementHistory:
    def __init__(self, ticker, fetch_date):
        self.ticker = ticker
        self.fetch_date = fetch_date
        self.quarter_statements = []  # array of Statements!
        self.year_statements = [] # array of Statements!
