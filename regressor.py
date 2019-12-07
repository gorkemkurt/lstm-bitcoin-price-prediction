from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


class Constants:
    NUM_UNITS = 10
    ACTIVATION_FUNCTION = 'sigmoid'
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'mean_squared_error'
    BATCH_SIZE = 10
    NUM_EPOCHS: int = 200


class Regressor:
    def __init__(self, x_train, y_train, x_validate, y_validate):
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate

    def train(self):
        # Initialize the RNN
        regressor = Sequential()
        # Adding the input layer and the LSTM layer
        regressor.add(LSTM(units=Constants.NUM_UNITS,
                           activation=Constants.ACTIVATION_FUNCTION,
                           input_shape=(None, 1)))
        # Adding the output layer
        regressor.add(Dense(units=1))
        # Compiling the RNN
        regressor.compile(optimizer=Constants.OPTIMIZER,
                          loss=Constants.LOSS_FUNCTION)
        # Using the training set to train the model
        regressor.fit(self.x_train,
                      self.y_train,
                      batch_size=Constants.BATCH_SIZE,
                      epochs=Constants.NUM_EPOCHS)
        result = regressor.evaluate(self.x_validate,
                                    self.y_validate,
                                    batch_size=2 * Constants.BATCH_SIZE)
        print(result)
        return regressor
