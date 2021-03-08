import os

from specusticc.configs_init.configer import Configer
from specusticc.market.market import Market


def test_runApplicationForMLP_sampleConfig_returnsCSVReport():
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


def test_runApplicationForCNN_sampleConfig_returnsCSVReport():
    model_names = ["cnn"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"cnn_1.csv", "cnn_2.csv", "true_data.csv"}.issubset(set(test_0_files)) and {
        "cnn_1.csv",
        "cnn_2.csv",
        "true_data.csv",
    }.issubset(set(train_files))


def test_runApplicationForLSTM_sampleConfig_returnsCSVReport():
    model_names = ["lstm"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"lstm_1.csv", "lstm_2.csv", "true_data.csv"}.issubset(
        set(test_0_files)
    ) and {"lstm_1.csv", "lstm_2.csv", "true_data.csv"}.issubset(set(train_files))


def test_runApplicationForCNNLSTM_sampleConfig_returnsCSVReport():
    model_names = ["cnn-lstm"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"cnn-lstm_1.csv", "cnn-lstm_2.csv", "true_data.csv"}.issubset(
        set(test_0_files)
    ) and {"cnn-lstm_1.csv", "cnn-lstm_2.csv", "true_data.csv"}.issubset(
        set(train_files)
    )


def test_runApplicationForLSTMAttention_sampleConfig_returnsCSVReport():
    model_names = ["lstm-attention"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"lstm-attention_1.csv", "lstm-attention_2.csv", "true_data.csv"}.issubset(
        set(test_0_files)
    ) and {"lstm-attention_1.csv", "lstm-attention_2.csv", "true_data.csv"}.issubset(
        set(train_files)
    )


def test_runApplicationForEncoderDecoder_sampleConfig_returnsCSVReport():
    model_names = ["encoder-decoder"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"encoder-decoder_1.csv", "encoder-decoder_2.csv", "true_data.csv"}.issubset(
        set(test_0_files)
    ) and {"encoder-decoder_1.csv", "encoder-decoder_2.csv", "true_data.csv"}.issubset(
        set(train_files)
    )


def test_runApplicationForTransformer_sampleConfig_returnsCSVReport():
    model_names = ["transformer"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"transformer_1.csv", "transformer_2.csv", "true_data.csv"}.issubset(
        set(test_0_files)
    ) and {"transformer_1.csv", "transformer_2.csv", "true_data.csv"}.issubset(
        set(train_files)
    )


def test_runApplicationForGAN_sampleConfig_returnsCSVReport():
    model_names = ["gan"]
    example_path = "./tests/e2e"

    configer = Configer(save_path=example_path)
    configer.create_all_configs()
    configs = configer.get_configs_wrapper()

    market = Market(configs=configs, models=model_names)
    market.run()

    # assert that all needed files are created
    test_0_files = os.listdir("./tests/e2e/test_0")
    train_files = os.listdir("./tests/e2e/train")
    assert {"gan_1.csv", "gan_2.csv", "true_data.csv"}.issubset(set(test_0_files)) and {
        "gan_1.csv",
        "gan_2.csv",
        "true_data.csv",
    }.issubset(set(train_files))
