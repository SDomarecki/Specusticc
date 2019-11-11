import pymongo
import pytest


@pytest.fixture()
def client():
    return pymongo.MongoClient("mongodb://localhost:27017/")


def test_dbms_connection(client):
    print(client.list_database_names())
    assert True


def test_collection_createandinsert(client):
    db = client["test-database"]
    collection = db['test-collection']

    import datetime
    post = {"author": "Mike",
            "text": "My first blog post!",
            "tags": ["mongodb", "python", "pymongo"],
            "date": datetime.datetime.utcnow()}

    posts = db.posts
    post_id = posts.insert_one(post).inserted_id
    print(post_id)
    print(db.list_collection_names())
    assert True


def test_collection_findall(client):
    db = client["stocks"]
    collection = db['companies']

    for company in collection.find():
        print(company)

    assert True
