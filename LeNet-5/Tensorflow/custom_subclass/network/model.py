import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class LeNet(Model):
    """ Custom LeNet-5 neural network model """

    def __init__(self, inputShape):
        super(LeNet, self).__init__()
        self.inputShape = inputShape
        # First block
        self.conv1 = Conv2D(6, (5, 5), padding='valid', activation='relu',
                            input_shape=self.inputShape, kernel_initializer='he_uniform')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Second block
        self.conv2 = Conv2D(16, (5, 5), padding='valid',
                            activation='relu', kernel_initializer='he_uniform')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Third block
        self.flatten = Flatten()
        self.dense1 = Dense(400, activation='relu',
                            kernel_initializer='he_uniform')
        self.dense2 = Dense(120, activation='relu',
                            kernel_initializer='he_uniform')
        self.dense3 = Dense(84, activation='relu',
                            kernel_initializer='he_uniform')

        # Output
        self.dense4 = Dense(10, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x = self.dense4(x)

        return x
