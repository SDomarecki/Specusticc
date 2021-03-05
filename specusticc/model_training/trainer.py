import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as C

from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.data_preprocessing.preprocessed_data import PreprocessedData
from specusticc.utilities.timer import Timer


class Trainer:
    def __init__(self, data: PreprocessedData, model_name: str, config: AgentConfig):
        self.data: PreprocessedData = data
        self.model_name = model_name
        self.config: AgentConfig = config

        self.save_fname = "temp.h5"
        self.callbacks = [
            # in all callbacks monitor='val_loss'
            C.EarlyStopping(mode="min", patience=40, verbose=1),
            C.ReduceLROnPlateau(factor=0.3, min_delta=0.01, patience=10, verbose=1),
            C.ModelCheckpoint(filepath=self.save_fname, save_best_only=True, verbose=1),
            # C.TensorBoard(),
            # C.CSVLogger(filename='learning.log')
        ]

    def train(self, model):
        X = self.data.train_set.get_input(self.model_name)
        Y = self.data.train_set.get_output()

        if self.model_name == "gan":
            return model.train(X, Y)

        t = Timer()
        t.start()

        epochs = self.config.epochs
        model.summary()
        history = model.fit(
            X,
            Y,
            epochs=epochs,
            shuffle=True,
            validation_split=0.1,
            callbacks=self.callbacks,
        )
        # model.save(self.save_fname)
        t.stop()
        t.print_time()

        self._plot_history(history)
        return model

    def _plot_history(self, history):
        plt.plot(history.history["mean_absolute_percentage_error"])
        if "val_mean_absolute_percentage_error" in history.history:
            plt.plot(history.history["val_mean_absolute_percentage_error"])
        plt.title("mean absolute percentage error")
        plt.ylabel("error")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.legend(["train", "validation"], loc="upper left")
        plt.grid()
        plt.show()
