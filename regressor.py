import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras import backend


class Constants:
    NUM_UNITS = 30
    ACTIVATION_FUNCTION = 'sigmoid'
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'mean_squared_error'
    BATCH_SIZE = 32
    NUM_EPOCHS = 200


class Regressor:
    def __init__(self, x_train, y_train, x_validate, y_validate):
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate

    def train(self):
        regressor = Sequential()
        regressor.add(LSTM(units=Constants.NUM_UNITS,
                           activation=Constants.ACTIVATION_FUNCTION,
                           input_shape=(30, 1)))
        regressor.add(Dense(units=5))

        # adamOpti = Adam(lr=0.0001)
        def rmse(y_true, y_pred):
            return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

        regressor.compile(optimizer=Constants.OPTIMIZER,
                          loss=Constants.LOSS_FUNCTION,
                          metrics=[rmse])
        history = regressor.fit(self.x_train,
                                self.y_train,
                                validation_data=(self.x_validate, self.y_validate),
                                batch_size=Constants.BATCH_SIZE,
                                epochs=Constants.NUM_EPOCHS,
                                verbose=2)

        regressor.evaluate(self.x_validate,
                           self.y_validate,
                           batch_size=Constants.BATCH_SIZE)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['rmse'])
        plt.plot(history.history['val_rmse'])
        plt.title('model rmse')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return regressor
