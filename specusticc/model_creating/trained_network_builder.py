from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.data_preprocessing.data_holder import DataHolder


class TrainedNetworkBuilder:
    def __init__(self, data: DataHolder, config: ModelCreatorConfig):
        self.data: DataHolder = data
        self.config: ModelCreatorConfig = config
        self.epochs: int = 20

    def build(self):
        self._choose_network()
        if self.config.optimize_hyperparameters:
            return self._optimize_network()
        else:
            return self._build_predefined_network()

    def _choose_network(self):
        from specusticc.model_creating.models.basic_net import BasicNet
        from specusticc.model_creating.models.optimized.cnn import CNN
        from specusticc.model_creating.models.lstm import LSTM
        from specusticc.model_creating.models.lstm_attention import LSTMAttention
        from specusticc.model_creating.models.mlp import MLP
        from specusticc.model_creating.models.lstm_enc_dec import LSTMEncoderDecoder
        from specusticc.model_creating.models.transformer import ModelTransformer

        name = self.config.name
        if name == 'basic':
            self.model = BasicNet(self.config)
        elif name == 'cnn':
            self.model = CNN(self.config)
        elif name == 'lstm':
            self.model = LSTM(self.config)
        elif name == 'lstm-attention':
            self.model = LSTMAttention(self.config)
        elif name == 'encoder-decoder':
            self.model = LSTMEncoderDecoder(self.config)
        elif name == 'mlp':
            self.model = MLP(self.config)
        elif name == 'transformer':
            self.model = ModelTransformer(self.config)
        else:
            raise NotImplementedError

    def _optimize_network(self):
        X = self.data.get_train_input()
        Y = self.data.get_train_output()
        build_fn = self.model.build_model
        possible_params = self.model.possible_parameters

        regressor = KerasRegressor(build_fn=build_fn, verbose=1)
        grid = GridSearchCV(estimator=regressor, param_grid=possible_params, n_jobs=1, cv=3)
        grid_result = grid.fit(X, Y)

        # summarize results
        print(f'Best: {-grid_result.best_score_} using {grid_result.best_params_}')
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (-mean, stdev, param))

        model = grid_result.best_estimator_.model
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
        return model

    def _build_predefined_network(self):
        return self.model.build_model()