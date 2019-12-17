import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from plotter import Plotter
from data_splitter import DataSplitter
from regressor import Regressor
from sklearn.metrics import mean_squared_error
from math import sqrt

min_max_scaler = MinMaxScaler()
df = pd.read_csv("market-price-2014.csv")
df_norm = df.drop(df.columns[0], 1, inplace=True)
data_splitter = DataSplitter(df)
df_train, df_validate, df_test = data_splitter.train_validate_test_split()

data_splitter = DataSplitter(df_train)
x_train, y_train = data_splitter.get_XY_sets(min_max_scaler, 30, 5)

data_splitter = DataSplitter(df_validate)
x_validate, y_validate = data_splitter.get_XY_sets(min_max_scaler, 30, 5)

regressor = Regressor(x_train, y_train, x_validate, y_validate).train()

# PREDICT PRICE
test_set = df_test.values
data_splitter = DataSplitter(df_test)
inputs, outputs = data_splitter.get_XY_sets(min_max_scaler, 30, 5)
predicted_price = regressor.predict(inputs)

x = np.array(outputs).ravel()
y = np.array(predicted_price).ravel()
rmse = sqrt(mean_squared_error(x, y))
print('RMSE: %.3f' % rmse)

predicted_price = min_max_scaler.inverse_transform(np.array(predicted_price[-2]).reshape(-1, 1)).tolist()
plotter = Plotter(test_set[-10:-5], predicted_price)
plotter.plot()
