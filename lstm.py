import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from plotter import Plotter
from data_splitter import DataSplitter
from regressor import Regressor

min_max_scaler = MinMaxScaler()
df = pd.read_csv("market-price-2010.csv")
df_norm = df.drop(df.columns[0], 1, inplace=True)
data_splitter = DataSplitter(df)
df_train, df_validate, df_test = data_splitter.train_validate_test_split()

data_splitter = DataSplitter(df_train)
x_train, y_train = data_splitter.get_XY_sets(min_max_scaler)

data_splitter = DataSplitter(df_validate)
x_validate, y_validate = data_splitter.get_XY_sets(min_max_scaler)

regressor = Regressor(x_train, y_train, x_validate, y_validate).train()

# PREDICT PRICE
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_price = regressor.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)

for i in range(len(inputs)):
    print("X=%s, Predicted=%s" % (test_set[i], predicted_price[i]))

plotter = Plotter(test_set, predicted_price)
plotter.plot()
