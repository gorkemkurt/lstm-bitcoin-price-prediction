import numpy as np


class DataSplitter:
    def __init__(self, df):
        self.df = df

    def train_validate_test_split(self):
        length = len(self.df)
        train = self.df[0:int(0.6 * length)]
        validate = self.df[int(0.6 * length):int(0.8 * length)]
        test = self.df[int(0.8 * length):]
        return train, validate, test

    def train_validate_test_split_random(self, train_percent=.6, validate_percent=.2, seed=None):
        # train, validate, test = np.split(self.df.sample(frac=1), [int(.6 * len(self.df)), int(.8 * len(self.df))])
        np.random.seed(seed)
        perm = np.random.permutation(self.df.index)
        m = len(self.df.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = self.df.iloc[perm[:train_end]]
        validate = self.df.iloc[perm[train_end:validate_end]]
        test = self.df.iloc[perm[validate_end:]]
        return train, validate, test

    def get_XY_sets(self, min_max_scaler):
        training_set = self.df.values
        training_set = min_max_scaler.fit_transform(training_set)
        x_set = training_set[0:len(training_set) - 1]
        y_set = training_set[1:len(training_set)]
        x_set = np.reshape(x_set, (len(x_set), 1, 1))
        return x_set, y_set
