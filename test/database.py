import pytest
import pymongo


@pytest.fixture()
def client():
    return pymongo.MongoClient("mongodb://localhost:27017/")


def test_database_connection(client):
    print(client.list_database_names())
    assert True

