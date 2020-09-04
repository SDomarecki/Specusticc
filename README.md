# Specusticc - Master Thesis Project

## About

## How it works

## How to run

Project developed on `Python 3.8.2.` with `PyCharm 2020.2`
All of the important run configs compatible with PyCharm are shared along with the code.

If you want to run in console or other IDE:
> - `requirements.txt` stores all of required libraries
> - `specusticc/specusticc_main.py` is the main runfile
> - run arguments are as follows: `<config_file> [<network name>]` <br>
i.e. `test2.json mlp cnn lstm transformer`.<br> 
List of available config files is stored in `raw_configs`. <br>
Available networks: `basic, mlp, cnn, lstm, cnn-lstm, encoder-decoder, lstm-attention, transformer, gan`
> - output of the execution is stored in `output/` directory. It contains runtime configuration and predictions with true data for each test set.

To replicate tests of master thesis run `specusticc_main.py` with following arguments:
- test 1 (no additional data): `test1_no_context_data.json mlp cnn lstm cnn-lstm encoder-decoder lstm-attention transformer gan`
- test 2 (correlated data): `test1_correlated_data.json mlp cnn lstm cnn-lstm encoder-decoder lstm-attention transformer gan`
- test 3 (non correlated data): `test1_non_correlated_data.json mlp cnn lstm cnn-lstm encoder-decoder lstm-attention transformer gan`
- test 4 (two test datasets, past&future): `test2.json lstm-attention`
- test 5 (horizon = 5): `test3_5.json lstm-attention`
- test 6 (horizon = 10): `test3_10.json lstm-attention`
- test 7 (horizon = 15): `test3_15.json lstm-attention`
- test 8 (horizon = 20): `test3_20.json lstm-attention`
- test 9 (horizon = 25): `test3_25.json lstm-attention`
- test 10 (horizon = 30): `test3_30.json lstm-attention`

## Visualization

Specusticc comes with its submodule to handle any of visualizations.
To run:
1. move result directory from `specusticc/output` to `specustic_visualizer/output`
2. in `visualizer_main.py` assign `{output_path}` variable to results directory, e.g `output/test`
3. run `visualizer_main.py`
Plots generated during this run will be shown in matplotlib window and saved in `{output_path}/plots`