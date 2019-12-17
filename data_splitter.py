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
        np.random.seed(seed)
        perm = np.random.permutation(self.df.index)
        m = len(self.df.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = self.df.iloc[perm[:train_end]]
        validate = self.df.iloc[perm[train_end:validate_end]]
        test = self.df.iloc[perm[validate_end:]]
        return train, validate, test

    def get_XY_sets(self, min_max_scaler, n_in, n_out):
        x, y = [], []
        training_set = self.df.values
        training_set = min_max_scaler.fit_transform(training_set)
        for i in range(len(training_set)):
            end = i + n_in
            out_end = end + n_out
            if out_end > len(training_set):
                break
            seq_x, seq_y = training_set[i:end], training_set[end:out_end]
            x.append(seq_x)
            y.append(seq_y)
        x = np.reshape(x, (len(x), n_in, 1))
        y = np.reshape(y, (len(y), n_out))
        return x, y

    def split_sequence(self, n_steps_in, n_steps_out):
        x, y = [], []
        for i in range(len(self.df)):
            end = i + n_steps_in
            out_end = end + n_steps_out
            if out_end > len(self.df):
                break
            seq_x, seq_y = self.df[i:end], self.df[end:out_end]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)
