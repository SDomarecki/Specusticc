import datetime
import os
import pandas as pd
import pymongo
from bs4 import BeautifulSoup
import json
from database.models.company import Company
from database.models.statement_history import StatementHistory


class DataMiner:
    def __init__(self):
        self.now = datetime.datetime.utcnow()
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.companies = self.read_companies_source()
        self.directories = self.get_ticker_list()

    @staticmethod
    def read_companies_source() -> [Company]:
        companies = []

        file_contents = DataMiner.load_file(filename='companies.json')
        data = json.loads(file_contents)
        for company in data:
            companies.append(Company(name=company['name'], ticker=company['ticker'], link=company['link']))
        return companies

    @staticmethod
    def get_ticker_list() -> [str]:
        listdir = os.listdir(os.curdir)
        listdir.remove("companies.json")
        return listdir

    def mine_and_save_all_companies(self):
        for company in self.companies:
            statement_history = self.fetch_company_fundamentals(company.ticker)
            self.save_statement_history_to_database(statement_history)

    def fetch_company_fundamentals(self, ticker) -> StatementHistory:
        statement_history = StatementHistory(ticker, self.now)

        quarterly_dataframes_list = []
        yearly_dataframes_list = []

        # TODO use the fucking fetch methods Johnny!

        quarterly_dataframe_merged = pd.concat(quarterly_dataframes_list, axis=1, sort=False).sort_index(inplace=False)
        yearly_dataframe_merged = pd.concat(yearly_dataframes_list, axis=1, sort=False).sort_index(inplace=False)

        statement_history.quarter_statements = self.transform_dataframe_into_list(quarterly_dataframe_merged)
        statement_history.year_statements = self.transform_dataframe_into_list(yearly_dataframe_merged)

        return statement_history

    def fetch_gain_and_loss(self, page):
        soup = BeautifulSoup(page, 'html.parser')
        if soup.find("table", class_="report-table") is None:
            return None
        table_rows = soup.find("table", class_="report-table").find_all("tr")

        seasons = table_rows[0].find_all("th")[1:-1]
        dates = []
        for season in seasons:
            dates.append(season.contents[0].strip())

        result_dataframe = pd.DataFrame()

        for row in table_rows[1:]:
            indicators = []
            tds = row.find_all("td")
            indicator_name = tds[0].text
            for td in tds[1:-1]:
                cleaned_td = self.trim_td(td.text)
                indicators.append(cleaned_td)
            result_dataframe[indicator_name] = indicators

        result_dataframe.index = dates

        return result_dataframe

    @staticmethod
    def trim_td(text: str) -> str:
        new_text = text.replace(" ", "")
        k_index = text.find('k')
        if k_index != -1:
            new_text = new_text[:k_index-1]
        r_index = text.find('r')
        if r_index != -1:
            new_text = new_text[:r_index-1]
        return new_text


    def save_statement_history_to_database(self, sh: StatementHistory):
        db = self.client['stocks']
        collection = db['statements']

        quarterlies = []
        yearlies = []

        for quarterly in sh.quarter_statements:
            quarterlies.append(quarterly.__dict__)
        for yearly in sh.year_statements:
            yearlies.append(yearly.__dict__)

        post = {
            'ticker': sh.ticker,
            'fetch_date': sh.fetch_date,
            'quarter_statements': quarterlies,
            'year_statements': yearlies
        }
        collection.insert_one(post)

    def save_to_database(self) -> None:
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
    def load_file(path='', filename='') -> str:
        file = open(path + filename, "r")
        page = file.read()
        file.close()
        return page


if __name__ == "__main__":
    os.chdir('html_sources')
    dm = DataMiner()

    res = dm.fetch_gain_and_loss(DataMiner.load_file('1AT/', 'statements_gain_and_loss_quarter.html'))
    print(res.to_string())