from main import Main
from data import loading_data


if __name__ == "__main__":
    train_ds, test_ds, input_shape = loading_data()

    nn = Main(input_shape=input_shape)
    nn.summary(output='model_summary', target='LeNet-5.txt')
    nn.compile(learning_rate=0.01, optimizer='sgd', loss='categorical_crossentropy', momentum=0.9)
    history = nn.fit(train_ds, validation_data = test_ds, epochs=1, batch_size = 32)
    nn.accuracy_graph(history)
    nn.loss_graph(history)
    nn.save('mnist.h5')

    

    
   