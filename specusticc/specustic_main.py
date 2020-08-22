import sys

import specusticc.utilities.directories as dirs
from specusticc.market.market import Market

if __name__ == "__main__":
    # timestamp = dirs.get_timestamp()
    # save_path = f'output/{timestamp}'

    # save_path = 'output/test'
    # save_path = 'output/test1_correlated'
    # save_path = 'output/test1_non_correlated'
    # save_path = 'output/test1_no_additional_data'
    save_path = 'output/test2'
    # save_path = 'output/test3_5'
    # save_path = 'output/test3_10'
    # save_path = 'output/test3_15'
    # save_path = 'output/test3_20'
    # save_path = 'output/test3_25'
    # save_path = 'output/test3_30'
    dirs.create_save_dir(save_path)

    config_file = sys.argv[1]
    model_names = sys.argv[2:]

    config_path = f'raw_configs/{config_file}'
    market = Market(config_path=config_path, models=model_names, save_path=save_path)
    market.run()
