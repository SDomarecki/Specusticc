import datetime
import pymongo
import requests
from bs4 import BeautifulSoup
import time

from models.company import Company

links = {
    'main': 'https://www.biznesradar.pl/',
    'stocks_list': 'gielda/akcje_gpw',
    'statements_gain_and_loss': 'raporty-finansowe-rachunek-zyskow-i-strat',
    'statements_balance': 'raporty-finansowe-bilans',
    'statements_cashflow': 'raporty-finansowe-przeplywy-pieniezne',
    'indicators_value': 'wskazniki-wartosci-rynkowej',
    'indicators_profitability': 'wskazniki-rentownosci',
    'indicators_cashflow': 'wskazniki-przeplywow-pienieznych',
    'indicators_debt': 'wskazniki-zadluzenia',
    'indicators_liquidity': 'wskazniki-plynnosci',
    'indicators_activity': 'wskazniki-aktywnosci'
}
class DatabaseCreate:
    def __init__(self):
        self.now = datetime.datetime.utcnow()
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.companies = []

    def fetch_company_list(self):
        url = links['main'] + links['stocks_list']
        page = self.fetch_page(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        for a in soup.find_all("a", class_="s_tt"):
            name = a['title']
            ticker = a.text.split(" ")[0]
            link = '/' + a['href'].split("/")[2]
            self.companies.append(Company(name, ticker, link))
            print(name)

    def save_to_database(self):
        db = self.client['stocks']
        collection = db['companies']
        fetched = 0
        for company in self.companies:
            post = {
                'name': company.name,
                'ticker': company.ticker,
                'link': company.link,
            }
            collection.insert_one(post)
            fetched += 1
            print(fetched)


    @staticmethod
    def fetch_page(url):
        page = ''
        while page == '':
            try:
                print('Connecting with: ' + url)
                page = requests.get(url)
                break
            except:
                print("Za szybki request, ponawiam polaczenie")
                time.sleep(5)
                continue
        print('Success')
        return page


if __name__ == "__main__":
    dbc = DatabaseCreate()
    dbc.fetch_company_list()
    dbc.save_to_database()