import json
from datetime import datetime
import os


def load_and_preprocess_config(config_path: str, backup_path: str) -> dict:
    config = _load_config(config_path)
    _backup_config(config, backup_path)
    config = _preprocess_config(config)
    return config


def _backup_config(config: dict, save_path: str):
    path = os.getcwd()
    full_save_path = path + '/' + save_path + '/config.json'
    with open(full_save_path, 'w') as outfile:
        json.dump(config, outfile)


def _load_config(config_path: str) -> dict:
    file = open(config_path)
    return json.load(file)


def _preprocess_config(config: dict) -> dict:
    config = _transform_datestring_to_datetime(config)
    return config


def _transform_datestring_to_datetime(config: dict) -> dict:
    date_format = '%Y-%m-%d'

    if 'train_date' in config['import']:
        import_train_date = config['import']['train_date']
        import_train_date['from'] = datetime.strptime(import_train_date['from'], date_format)
        import_train_date['to'] = datetime.strptime(import_train_date['to'], date_format)
        config['import']['train_date'] = import_train_date

    if 'test_date' in config['import']:
        import_test_date = config['import']['test_date']
        import_test_date['from'] = datetime.strptime(import_test_date['from'], date_format)
        import_test_date['to'] = datetime.strptime(import_test_date['to'], date_format)
        config['import']['test_date'] = import_test_date
    return config
