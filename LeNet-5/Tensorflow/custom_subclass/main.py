import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from network.model import LeNet


class Main:
    def __init__(self, inputShape):
        self.inputShape = inputShape
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='test_accuracy')
        
        self.model = LeNet(self.inputShape)

    @tf.function
    def train_step(self, inputs, outputs):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss(outputs, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(outputs, predictions)

    @tf.function
    def test_step(self, inputs, outputs):
        predictions = self.model(inputs)
        t_loss = self.loss(outputs, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(outputs, predictions)

    def train(self, epochs=1, train_ds=None, test_ds=None):
        # Tensorboard Setup
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(epochs):
            for inputs, outputs in train_ds:
                self.train_step(inputs, outputs)
                print(f'EPOCHS = {epoch}')
                print(f'Training_Loss = {self.train_loss.result()}-----Training_Accuracy = {self.train_accuracy.result() * 100 } %')
                print("")
                print("")
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)


            for test_inputs, test_outputs in test_ds:
                self.test_step(test_inputs, test_outputs)
                print(f'EPOCHS = {epoch}')
                print(f'Testing_Loss = {self.test_loss.result()}-----Testing_Accuracy = {self.test_accuracy.result() * 100 } %')
                print("")
                print("")
                

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy:{}'
            print(template.format(epoch+1, self.train_loss.result(), self.train_accuracy.result()
                                  * 100, self.test_loss.result(), self.test_accuracy.result()*100))
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def save(self, model_name):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print('Model Saving')
        self.model.save(os.path.join(model_path, model_name))
        print('Model Saved !')
