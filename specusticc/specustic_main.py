import sys

from specusticc.configs_init.configer import Configer
from specusticc.market.market import Market

if __name__ == "__main__":
    example_directory = sys.argv[1]
    model_names = sys.argv[2:]

    example_path = f"./examples/{example_directory}"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()
