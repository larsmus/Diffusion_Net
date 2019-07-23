from keras.layers import Dense, Input, Flatten, Lambda
from keras.models import Model, Sequential
from keras import regularizers
import matplotlib.pyplot as plt


class Encoder:
    def __init__(self, reg=0., embedding_size=32):
        model = Sequential()
        # model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        model.add(Dense(embedding_size, activation='linear', kernel_regularizer=regularizers.l2(reg)))
        self.model = model

    def compile(self, optimizer="adam", loss="mean_squared_error"):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, x_train, train_embed, batch_size, epochs=100, x_test=None, test_embed=None):
        if x_test is None and test_embed is None:
            self.history = self.model.fit(x_train, train_embed, batch_size=batch_size, epochs=epochs, verbose=1,
                                          shuffle=True)
        elif x_test is not None and test_embed is not None:
            self.history = self.model.fit(x_train, train_embed, batch_size=batch_size, epochs=epochs, verbose=1,
                                          shuffle=True, validation_data=(x_test, test_embed))
        else:
            raise Exception("Invalid input data")

    def plot_progress(self):
        fig = plt.figure()

        # Plot training & validation loss values
        plt.subplot(1, 1, 1)
        plt.plot(self.history.history['loss'])
        plt.title(f'Model loss. Final train: {self.history.history["loss"][-1]}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper right')
        plt.show()


class Decoder:
    def __init__(self, input_size, reg=0.):
        model = Sequential()
        # model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        model.add(Dense(input_size, activation='linear', kernel_regularizer=regularizers.l1(reg)))
        self.model = model

    def compile(self, optimizer="adam", loss="mean_squared_error"):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, train_embed, x_train, batch_size, epochs=100):
        self.history = self.model.fit(train_embed, x_train, batch_size=batch_size, epochs=epochs, verbose=0,
                                      shuffle=True)

    def plot_progress(self):
        fig = plt.figure()

        # Plot training & validation loss values
        plt.subplot(1, 1, 1)
        plt.plot(self.history.history['loss'])
        plt.title(f'Model loss. Final train: {self.history.history["loss"][-1]}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper right')
        plt.show()


class Classifier:
    def __init__(self, reg=0):
        model = Sequential()
        # model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        # BatchNormalization()
        # model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        self.model = model

    def compile(self, optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, x_test, y_test, batch_size, epochs=100):
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                                      validation_data=(x_test, y_test), shuffle=True)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def plot_progress(self):
        fig = plt.figure()
        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()