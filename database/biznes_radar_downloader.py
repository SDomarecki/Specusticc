import json
import os
import time

import requests
from bs4 import BeautifulSoup

from models.company import Company

main_url = 'https://www.biznesradar.pl/'
stocks_list_url = 'gielda/akcje_gpw'

links = {
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


class BiznesRadarDownloader:
    def __init__(self) -> None:
        self.companies = []

    def fetch_companies_list(self) -> None:
        url = main_url + stocks_list_url
        page = self.fetch_page(url)
        soup = BeautifulSoup(page, 'html.parser')
        for a in soup.find_all("a", class_="s_tt"):
            name = a['title']
            ticker = a.text.split(" ")[0]
            link = '/' + a['href'].split("/")[2]
            self.companies.append(Company(name, ticker, link))

    def serialize_companies_list(self) -> None:
        s = json.dumps([company.__dict__ for company in self.companies])
        BiznesRadarDownloader.save_page_to_file(s, 'companies.json')

    def fetch_all_companies_pages(self) -> None:
        companies_to_fetch = len(self.companies)
        print('Companies to fetch: ' + str(companies_to_fetch))
        company_count = 0
        for company in self.companies:
            self.fetch_related_pages(company.ticker, company.link)
            company_count += 1
            print('Already fetched companies: [' + str(company_count) + '/' + str(companies_to_fetch) + ']')

    @staticmethod
    def fetch_related_pages(ticker, link) -> None:
        directory_created = BiznesRadarDownloader.create_directory(ticker)
        if not directory_created:
            return

        for page_name in links:
            url = main_url + links[page_name] + link
            page_content = BiznesRadarDownloader.fetch_page(url)
            save_path = ticker + '/' + page_name + '_year.html'
            BiznesRadarDownloader.save_page_to_file(page_content, save_path)

            url = main_url + links[page_name] + link + ',Q'
            page_content = BiznesRadarDownloader.fetch_page(url)
            save_path = ticker + '/' + page_name + '_quarter.html'
            BiznesRadarDownloader.save_page_to_file(page_content, save_path)

    @staticmethod
    def fetch_page(url) -> str:
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
        return page.text

    @staticmethod
    def create_directory(path) -> bool:
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
            return False
        else:
            print("Successfully created the directory %s " % path)
            return True

    @staticmethod
    def save_page_to_file(page, path) -> None:
        file = open(path, "w")
        file.write(page)
        file.close()


if __name__ == "__main__":
    os.chdir('html_sources')
    brd = BiznesRadarDownloader()
    brd.fetch_companies_list()
    brd.serialize_companies_list()
    brd.fetch_all_companies_pages()
