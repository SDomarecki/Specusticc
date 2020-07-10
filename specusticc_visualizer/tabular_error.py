import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TabularError:
    def count_errors(self, true_data: pd.DataFrame, prediction_data: dict):
        true_data = true_data.drop(columns=['date'])
        print('name, mape, mse, mae, r^2')
        for model_name, model_preds in prediction_data.items():
            mean_errors = self.count_one_model_errors(true_data, model_preds)
            print(f'{model_name}: {mean_errors}')

    def count_one_model_errors(self, true_data: pd.DataFrame, one_model_predictions: pd.DataFrame):
        one_model_predictions = one_model_predictions.drop(columns=['date'])
        columns = one_model_predictions.columns
        errors = np.array([])
        for column in columns:
            one_prediction = one_model_predictions[[column]]
            errors = np.append(errors, self.count_one_prediction_errors(true_data, one_prediction), axis=0)
        errors = errors.reshape((len(columns), -1))
        mean_errors = np.mean(errors, axis=0)
        return mean_errors

    def count_one_prediction_errors(self, true_data: pd.DataFrame, one_prediction: pd.DataFrame) -> np.array:
        mape = self.mean_absolute_percentage_error(true_data, one_prediction)
        mse = mean_squared_error(true_data, one_prediction)
        mae = mean_absolute_error(true_data, one_prediction)
        r2 = r2_score(true_data, one_prediction)

        errors = np.array([mape, mse, mae, r2])
        return errors

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

