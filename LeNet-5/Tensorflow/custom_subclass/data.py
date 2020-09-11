import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def loading_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Data Plotting
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()

    # add a dimension for channel
    x_train = x_train[:, :, :, tf.newaxis]
    x_test = x_test[:, :, :, tf.newaxis]
    # one hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    input_shape = x_train[0].shape
    return train_ds, test_ds, input_shape
