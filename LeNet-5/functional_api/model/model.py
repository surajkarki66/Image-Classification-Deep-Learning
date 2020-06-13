import tensorflow as tf


def LeNet5(input_shape=None):
    """ Building LeNet-5 Model using Functional API """
    input_data = tf.keras.layers.Input(shape=input_shape)
    # First block
    conv1 = tf.keras.layers.Conv2D(
        6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_uniform')(input_data)
    maxpool1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv1)

    # Second block
    conv2 = tf.keras.layers.Conv2D(16, (5, 5), padding='valid',
                                   activation='relu', kernel_initializer='he_uniform')(maxpool1)
    maxpool2 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv2)

    # Third block
    flatten = tf.keras.layers.Flatten()(maxpool2)
    dense1 = tf.keras.layers.Dense(400, activation='relu',
                                   kernel_initializer='he_uniform')(flatten)
    dense2 = tf.keras.layers.Dense(120, activation='relu',
                                   kernel_initializer='he_uniform')(dense1)
    dense3 = tf.keras.layers.Dense(84, activation='relu',
                                   kernel_initializer='he_uniform')(dense2)

    # Output
    dense4 = tf.keras.layers.Dense(10, activation='softmax')(dense3)

    model = tf.keras.models.Model(inputs=input_data, outputs=dense4)

    return model
