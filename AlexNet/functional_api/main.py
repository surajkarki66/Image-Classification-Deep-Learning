import os
import tensorflow as tf
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm


from model.model import AlexNet


class Main:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = AlexNet(input_shape=self.input_shape, num_classes=10)

    def summary(self, output=None, target=None):
        """ Show / Save model structure (summary) """

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):
        """ Load a model with checkpoint file"""

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target)

    def get_callbacks(self, logdir = None, checkpoint = None, monitor="val_loss", verbose=0):
        """Setup the list of callbacks for the model
        @param logdir:
        @param checkpoint:
        @param monitor:
        @param verbose:
        @return:
        """

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=20,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=15,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=0.001, optimizer=None, loss=None, momentum=None):
        """
        Configures the Model for training/predict.

        :param optimizer: optimizer for training
        @param learning_rate:
        """
        if optimizer == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate)

        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=momentum)

        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=['accuracy'])

        return self.model

    def fit(self, x=None, y=None, batch_size=None, epochs=1,
            verbose=1, callbacks=None, validation_split=0.0,
            validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_freq=1, max_queue_size=10,
            workers=1, use_multiprocessing=False, **kwargs):
        """
            Model training on data yielded (fit function has support to generator).
            A fit() abstration function of TensorFlow 2.

            Provide x parameter of the form: yielding (x, y, sample_weight).

            :param: See tensorflow.keras.Model.fit()
            :return: A history object
            """

        history = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 callbacks=callbacks, validation_split=validation_split,
                                 validation_data=validation_data, shuffle=shuffle,
                                 class_weight=class_weight, sample_weight=sample_weight,
                                 initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps, validation_freq=validation_freq,
                                 max_queue_size=max_queue_size, workers=workers,
                                 use_multiprocessing=use_multiprocessing, **kwargs)
        return history

    def accuracy_graph(self, history):
        plot_dir = "graph_plots/accuracy/"
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plt.savefig(os.path.join(plot_dir, "model_accuracy.png"))

    def loss_graph(self, history):
        plot_dir = "graph_plots/loss/"
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plt.savefig(os.path.join(plot_dir, "model_loss.png"))

    def save(self, model_name):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print('Model Saving')
        self.model.save(os.path.join(model_path, model_name))
        print('Model Saved !')

