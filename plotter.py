from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, test_set, predict_set):
        self.test_set = test_set
        self.predict_set = predict_set

    def plot(self):
        print("Predicted Prices:\n", self.predict_set)
        plt.plot(self.predict_set, color='blue', label='Predicted BTC Price')
        print("\nReal Prices:\n", self.test_set.tolist())
        plt.plot(self.test_set.tolist(), label='Real BTC Price')
        plt.title('BTC Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('BTC Price(USD)')
        plt.legend(loc='best')
        plt.show()
