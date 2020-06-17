from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from sklearn.model_selection import GridSearchCV
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.hyperparameter_optimization.gan import GAN


class Optimizer:
    def __init__(self, data: DataHolder, config: ModelCreatorConfig):
        self.data = data
        self.config = config
        self.epochs = 20

        self.input_timesteps = config.input_timesteps
        self.input_features = config.input_features
        self.output_timesteps = config.output_timesteps

    def optimize(self):
        X = self.data.get_train_input()
        Y = self.data.get_train_output()

        gan = GAN(self.config)
        gan.train(X, Y)

    def old_optimize(self):
        X = self.data.get_train_input()
        Y = self.data.get_train_output()

        # create model
        model = KerasRegressor(build_fn=self.create_model, verbose=1)
        # define the grid search parameters
        # batch_size = [10, 20, 50, 100, 200, 500]
        # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        # neurons = [20, 50, 100, 200]
        # activation = ['softmax', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']

        batch_size = [10, 50]
        optimizer = ['Adam', 'Nadam']
        neurons = [20, 100]
        activation = ['relu', 'sigmoid', 'linear']
        dropout_rate = [0.0, 0.2, 0.4, 0.6, 0.8]

        param_grid = dict(batch_size=batch_size, dropout_rate=dropout_rate, optimizer=optimizer, neurons=neurons, activation=activation)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
        grid_result = grid.fit(X, Y)
        # summarize results
        print("Best: %f using %s" % (-grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (-mean, stdev, param))

    def create_model(self, optimizer='adam', dropout_rate=0.0, neurons=20, activation='relu'):
        model = M.Sequential()

        model.add(L.Dense(units=neurons, input_dim=self.input_timesteps*self.input_features, activation=activation))
        model.add(L.Dropout(dropout_rate))
        model.add(L.Dense(units=neurons, activation=activation))
        model.add(L.Dropout(dropout_rate))
        model.add(L.Dense(units=neurons, activation=activation))
        model.add(L.Dense(self.output_timesteps, activation='linear'))

        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"])
        return model