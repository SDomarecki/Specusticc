from model_creating.choose_network import choose_network
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from configs_init.model.agent_config import AgentConfig
from data_preprocessing.preprocessed_data import PreprocessedData

class Optimizer:
    def __init__(self, data: PreprocessedData, model_name: str, config: AgentConfig):
        self.data: PreprocessedData = data
        self.model_name = model_name
        self.model_builder = None
        self.config: AgentConfig = config

    def optimize(self):
        self.model_builder = choose_network(self.model_name, self.config)
        return self._grid_optimization()

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
