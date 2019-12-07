from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, test_set, predict_set):
        self.test_set = test_set
        self.predict_set = predict_set

    def plot(self):
        plt.figure(figsize=(25, 25), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(self.test_set[:, 0], color='red', label='Real BTC Price')
        plt.plot(self.predict_set[:, 0], color='blue', label='Predicted BTC Price')
        plt.title('BTC Price Prediction', fontsize=40)
        plt.xlabel('Time', fontsize=40)
        plt.ylabel('BTC Price(USD)', fontsize=40)
        plt.legend(loc='best')
        plt.show()
