[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![Updates](https://pyup.io/repos/github/SDomarecki/Specusticc/shield.svg)](https://pyup.io/repos/github/SDomarecki/Specusticc/)
[![Python application](https://github.com/SDomarecki/Specusticc/workflows/Python%20application/badge.svg)](https://github.com/SDomarecki/Specusticc/actions)
[![codecov](https://codecov.io/gh/SDomarecki/Specusticc/branch/master/graph/badge.svg?token=9TL9ND6579)](https://codecov.io/gh/SDomarecki/Specusticc)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Specusticc

Specusticc utilizes neural networks in prediction of stock prices of companies listed on Warsaw Stock Exchange.
Dataset is split into samples with rolling window.
Then samples are used to predict all of the values in horizon.
Predictions can be visualized in separate script `visualization_main.py`.

Repository comes with sample data to replicate all of the examples.
For more: download charts from [Stooq.pl](https://www.stooq.pl) by hand or via bundled script.

Specusticc uses various neural net architectures such as:
- MLP
- CNN
- LSTM
- CNN-LSTM
- Encoder-decoder
- Encoder-decoder with Attention mechanism
- Transformer (as in Attention is all you need)
- GAN

## Usage

#### Regular run
Main part of the application can be executed by:
```
$ python specusticc\\specustic_main.py {test_directory} [networks]
```
where {test_directory} must have a config.json file and networks is a list of models to use,
e.g.
```
$ python specusticc\\specustic_main.py regular_test mlp cnn lstm
```

Available networks: `basic, mlp, cnn, lstm, cnn-lstm, encoder-decoder, lstm-attention, transformer, gan`.

#### Visualization
To draw plots of prediction:
- point `results_path` in `specusticc_visualizer/visualizer_main.py` to valid results,
- run
```
$ python specusticc_visualizer\visualizer_main.py
```
To reduce noise on plots in grand tests (with all of the architectures) only 3 networks at a time are drawn with full opacity.
See `summary_plotter.py` to change its behaviour to suit your needs.

#### Config.json
List of all of the options:
```bash
"import": {
    "datasource": "csv", #available 'mongodb' or 'csv'
    "input": { #input of neural network
      "tickers": [ 
        "PKO"
      ],
      "columns": [ #OHLCV or OHLC for indexes
        "open", "high", "low", "close", "vol"
      ]
    },
    "target": { #output of neural network
      "tickers": [
        "PKO"
      ],
      "columns": [ #OHLCV
        "close"
      ]
    },
    "context": { #additional info, 
                 #used mainly in encoder-decoder architectures, 
                 #can be ommited
      "tickers": [
        "PZU", "ALR", "BHW",
        "BOS", "ING", "MBK",
        "MIL", "PEO", "WIG",
        "^SPX"
      ],
      "columns": [ #OHLCV
        "open", "high", "low", "close", "vol"
      ]
    },
    "train_date": { #RRRR-MM-DD
      "from": "2004-01-01", "to": "2017-12-31"
    },
    "test_date": [ #RRRR-MM-DD, multiple data ranges are handled
      {"from": "2018-01-01", "to": "2019-06-30"}
    ]
  },
  "preprocessing": {
    "window_length": 200, #length of one sample
    "rolling": 5, #subsample shift
    "horizon": 5 #prediction horizon, Specusticc will predict all of the values 
                 #from end_of_sample to end_of_sample+horizon
  },
  "agent": {
    "folds": 10, #number of separate folds to compute mean error
    "hyperparam_optimization_method": "none" #atm. "grid" is supported for grid search
                                             #if used, set folds to 1, otherwise it will optimize folds times
  }
```

## Install

To clone and run this application, you'll need [Git](https://git-scm.com) and [Python interpreter](https://www.python.org/downloads/)
From your command line:

```bash
# Clone this repository
$ git clone https://github.com/SDomarecki/Specusticc

# Go into the repository
$ cd Specusticc

# (optional) Install virtualenv for venv handling
$ pip install virtualenv

# (optional) Create virtual environment for this project
$ virtualenv venv

# (optional) Activate new venv
$ venv/Scripts/activate

# Install dependencies
$ pip install -r requirements.txt

# Run script with any sample config
$ python specusticc\\specustic_main.py regular_test lstm-attention
```
Aaand it's ready to go.
Alternative way is to use bundled `PyCharm` configurations.

## See Also

- [Clairvoyant](https://github.com/anfederico/Clairvoyant) -
Supervised learning algorithm using financial data. Focuses mainly on SVM.
- [bulbea](https://github.com/achillesrasquinha/bulbea) - Stock market prediction with deep learning models.

## License

Project created as base for master thesis "Application of machine learning methods in prediction of stock exchange prices" written @ AGH UST 2020.

This project is released under the MIT Licence. See the bundled LICENSE file for details.

(c) S.Domarecki 2019
