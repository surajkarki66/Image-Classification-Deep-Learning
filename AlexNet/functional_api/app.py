import os
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from data import loading_data
from main import Main

if __name__ == "__main__":
    train_ds, test_ds, input_shape = loading_data()
    m = Main(input_shape=input_shape)
    callbacks = [m.get_callbacks(logdir='tensorboard')[1]]
    m.summary()
    m.compile(learning_rate=0.01, optimizer='sgd',
              loss='categorical_crossentropy', momentum=0.9)
    history = m.fit(train_ds,
                    epochs=1, batch_size=64, callbacks=callbacks, validation_data=test_ds)
    m.accuracy_graph(history)
    m.loss_graph(history)
