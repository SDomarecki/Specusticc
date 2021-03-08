import os

from specusticc.configs_init.configer import Configer
from specusticc.market.market import Market


def test_runApplication_sampleConfig_returnsCSVReport():
    model_names = ["mlp"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"mlp_1.csv", "mlp_2.csv", "true_data.csv"}.issubset(set(test_0_files)) and {
        "mlp_1.csv",
        "mlp_2.csv",
        "true_data.csv",
    }.issubset(set(train_files))
