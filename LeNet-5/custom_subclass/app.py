import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from main import Main
from data  import loading_data

if __name__ == "__main__":
    train_ds, test_ds, input_shape = loading_data() 
    m = Main(input_shape)
    m.train(train_ds = train_ds, test_ds = test_ds, epochs=1)
    m.save('mnist_epochs10.tf')

