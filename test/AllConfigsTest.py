import pytest
from specusticc.agent import Agent

config_paths = [
    'configs/lstm_regression.json',
    'configs/lstm_classification.json',
    'configs/cnn_regression.json',
    'configs/cnn_classification.json',
    'configs/tree_classification.json'
]


def test_lstm_regression():
    config_path = config_paths[0]
    agent = Agent(config_path=config_path)
    assert True


def test_lstm_classification():
    config_path = config_paths[1]
    agent = Agent(config_path=config_path)
    assert True


def test_cnn_regression():
    config_path = config_paths[2]
    agent = Agent(config_path=config_path)
    assert True


def test_cnn_classification():
    config_path = config_paths[3]
    agent = Agent(config_path=config_path)
    assert True


def test_tree_classification():
    config_path = config_paths[4]
    agent = Agent(config_path=config_path)
    assert True
