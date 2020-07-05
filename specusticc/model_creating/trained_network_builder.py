from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt

from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.utilities.timer import Timer
import tensorflow.keras.callbacks as C

class TrainedNetworkBuilder:
    def __init__(self, data: DataHolder, model_name: str, config: AgentConfig):
        self.data: DataHolder = data
        self.model_name = model_name
        self.config: AgentConfig = config
        self.epochs: int = 20

    def build(self):
        self._choose_network()
        if self.config.hyperparam_optimization_method in ['grid', 'random', 'bayes']:
            return self._optimize_network()
        else:
            return self._build_and_train_predefined_network()

    def _choose_network(self):
        from specusticc.model_creating.models.basic_net import BasicNet
        from specusticc.model_creating.models.cnn import CNN
        from specusticc.model_creating.models.lstm import LSTM
        from specusticc.model_creating.models.lstm_attention import LSTMAttention
        from specusticc.model_creating.models.mlp import MLP
        from specusticc.model_creating.models.lstm_enc_dec import LSTMEncoderDecoder
        from specusticc.model_creating.models.transformer import ModelTransformer

        name = self.model_name
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
        method = self.config.hyperparam_optimization_method
        if method == 'grid':
            return self._grid_optimization()
        elif method == 'random':
            return self._random_optimization()
        else:
            return self._bayes_optimization()

    def _grid_optimization(self):
        X = self.data.get_train_input(self.model_name)
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

    def _random_optimization(self):
        raise NotImplementedError

    def _bayes_optimization(self):
        raise NotImplementedError

    def _build_and_train_predefined_network(self):
        seq = self.model.build_model()
        seq = self._train(seq)
        return seq

    def _build_predefined_network(self):
        return self.model.build_model()

    def _train(self, seq):
        print('[Model] Training Started')
        t = Timer()
        t.start()

        X = self.data.get_train_input(self.model_name)
        Y = self.data.get_train_output()

        save_fname = 'temp.h5'
        callbacks = [
            C.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20),
            C.ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.01, patience=5, verbose=1),
            C.ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True, verbose=1),
            # C.TensorBoard(),
            # C.CSVLogger(filename='learning.log')
        ]

        seq.summary()
        history = seq.fit(
                X,
                Y,
                epochs=self.epochs,
                callbacks=callbacks
        )
        self.save(seq, save_fname)
        print('[Model] Training Completed')
        t.stop()
        t.print_time()

        self._plot_history(history)

        return seq

    def save(self, seq, save_dir: str):
        seq.save(save_dir)

    def _plot_history(self, history):
        # summarize history for accuracy
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('mean squared error')
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid()
        plt.show()