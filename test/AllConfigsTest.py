from pathlib import Path

from specusticc.agent import Agent

config_paths = [
    'configs_bak/lstm_regression.json',
    'configs_bak/lstm_classification.json',
    'configs_bak/cnn_regression.json',
    'configs_bak/cnn_classification.json',
    'configs_bak/tree_classification.json'
]

# def test_lstm_regression():
#     config_path = config_path = Path(__file__).parent.parent / 'specusticc' / config_paths[0]
#     agent = Agent(config_path=config_path)
#     assert True


def test_lstm_classification():
    config_path = Path(__file__).parent.parent / 'specusticc' / config_paths[1]
    agent = Agent(config_path=str(config_path))
    assert True


# def test_cnn_regression():
#     config_path = config_path = Path(__file__).parent.parent / 'specusticc' / config_paths[2]
#     agent = Agent(config_path=str(config_path))
#     assert True
#
#
# def test_cnn_classification():
#     config_path = config_path = Path(__file__).parent.parent / 'specusticc' / config_paths[3]
#     agent = Agent(config_path=str(config_path))
#     assert True


def test_tree_classification():
    config_path = config_path = Path(__file__).parent.parent / 'specusticc' / config_paths[4]
    agent = Agent(config_path=str(config_path))
    assert True
