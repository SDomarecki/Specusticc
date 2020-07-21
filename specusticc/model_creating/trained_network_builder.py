import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as C
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData
from specusticc.utilities.timer import Timer


class TrainedNetworkBuilder:
    def __init__(self, data: PreprocessedData, model_name: str, config: AgentConfig):
        self.data: PreprocessedData = data
        self.model_name = model_name
        self.model_builder = None
        self.config: AgentConfig = config

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
        from specusticc.model_creating.models.cnnlstm import CNNLSTM
        from specusticc.model_creating.models.lstm_attention import LSTMAttention
        from specusticc.model_creating.models.mlp import MLP
        from specusticc.model_creating.models.lstm_enc_dec import LSTMEncoderDecoder
        from specusticc.model_creating.models.transformer import ModelTransformer
        from specusticc.model_creating.models.gan_wrapper import GANWrapper

        name = self.model_name
        if name == 'basic':
            self.model_builder = BasicNet(self.config)
        elif name == 'cnn':
            self.model_builder = CNN(self.config)
        elif name == 'lstm':
            self.model_builder = LSTM(self.config)
        elif name == 'cnn-lstm':

            self.model_builder = CNNLSTM(self.config)
        elif name == 'lstm-attention':
            self.model_builder = LSTMAttention(self.config)
        elif name == 'encoder-decoder':
            self.model_builder = LSTMEncoderDecoder(self.config)
        elif name == 'mlp':
            self.model_builder = MLP(self.config)
        elif name == 'transformer':
            self.model_builder = ModelTransformer(self.config)
        elif name == 'gan':
            self.model_builder = GANWrapper(self.config)
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
        X = self.data.train_set.get_input(self.model_name)
        Y = self.data.train_set.get_output()
        build_fn = self.model_builder.build_model
        possible_params = self.model_builder.possible_parameters

        regressor = KerasRegressor(build_fn=build_fn, verbose=1)
        grid = GridSearchCV(estimator=regressor, param_grid=possible_params, n_jobs=1, cv=10)
        grid_result = grid.fit(X, Y)

        # summarize results
        print(f'Best: {-grid_result.best_score_} using {grid_result.best_params_}')
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (-mean, stdev, param))

        model = grid_result.best_estimator_.model
        mape = 'mean_absolute_percentage_error'
        model.compile(loss=mape, optimizer="adam", metrics=[mape])
        return model

    def _random_optimization(self):
        raise NotImplementedError

    def _bayes_optimization(self):
        raise NotImplementedError

    def _build_and_train_predefined_network(self):
        model = self.model_builder.build_model()
        if self.model_name == 'gan':
            X = self.data.train_set.get_input(self.model_name)
            Y = self.data.train_set.get_output()
            return model.train(X, Y)

        model = self._train(model)
        return model

    def _train(self, model):
        print('[Model] Training Started')
        t = Timer()
        t.start()

        epochs = self.model_builder.epochs
        X = self.data.train_set.get_input(self.model_name)
        Y = self.data.train_set.get_output()

        save_fname = 'temp.h5'
        callbacks = [
            # C.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20),
            C.ReduceLROnPlateau(monitor='loss', factor=0.3, min_delta=0.01, patience=10, verbose=1),
            C.ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True, verbose=1),
            # C.TensorBoard(),
            # C.CSVLogger(filename='learning.log')
        ]

        model.summary()
        history = model.fit(
                X,
                Y,
                epochs=epochs,
                shuffle=True,
                validation_split=0.1,
                callbacks=callbacks
        )
        model.save(save_fname)
        print('[Model] Training Completed')
        t.stop()
        t.print_time()

        self._plot_history(history)

        return model

    def _plot_history(self, history):
        # summarize history for accuracy
        plt.plot(history.history['mean_absolute_percentage_error'])
        if 'val_mean_absolute_percentage_error' in history.history:
            plt.plot(history.history['val_mean_absolute_percentage_error'])
        plt.title('mean absolute percentage error')
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid()
        plt.show()
