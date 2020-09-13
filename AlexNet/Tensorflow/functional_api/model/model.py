import tensorflow as tf

def AlexNet(input_shape = None, num_classes = None):
    if input_shape and num_classes is not None:
        input_data = tf.keras.layers.Input(shape=input_shape)
        # First convolution block
        conv1 = tf.keras.layers.Conv2D(filters=96,
                                kernel_size=(11, 11),
                                strides=4,
                                padding="valid",
                                activation=tf.keras.activations.relu,
                                input_shape=(28, 28, 1))(input_data)
        maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=2,
                                    padding="valid")(conv1)
        batch_norm1 = tf.keras.layers.BatchNormalization()(maxpool1)

        # Second Convolutional Layer.
        conv2 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=(5, 5),
                                strides=1,
                                padding="same",
                                activation=tf.keras.activations.relu)(batch_norm1)
        maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=2,
                                    padding="same")(conv2)
        batch_norm2 = tf.keras.layers.BatchNormalization()(maxpool2)

        # Third Convolutional Layer.
        conv3 = tf.keras.layers.Conv2D(filters=384,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same",
                                activation=tf.keras.activations.relu)(batch_norm2)
        # Fourth Convolutional layer.
        conv4 = tf.keras.layers.Conv2D(filters=384,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same",
                                activation=tf.keras.activations.relu)(conv3)
        # Fifth convolutional layer.
        conv5 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same",
                                activation=tf.keras.activations.relu)(conv4)
        maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=2,
                                    padding="same")(conv5)
        batch_norm3 = tf.keras.layers.BatchNormalization()(maxpool3)

        # First fully connected layer
        flatten = tf.keras.layers.Flatten()(batch_norm3)
        fc1 = tf.keras.layers.Dense(units=4096,
                                activation=tf.keras.activations.relu)(flatten)
        drop1 = tf.keras.layers.Dropout(rate=0.2)(fc1)

        # Second fully connected layer.
        fc2 = tf.keras.layers.Dense(units=4096,
                                activation=tf.keras.activations.relu)(drop1)
        drop2 = tf.keras.layers.Dropout(rate=0.2)(fc2)

        # Output layer.
        out = tf.keras.layers.Dense(units=10,
                                activation=tf.keras.activations.softmax)(drop2)

        model = tf.keras.models.Model(inputs=input_data, outputs=out)
        model.summary()
        
        return model
print(AlexNet(input_shape=(28, 28, 1), num_classes=10))
