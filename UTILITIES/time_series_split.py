#!/usr/bin/env python
# coding: utf-8

# In[3]:
import pandas as pd
import numpy as np

class OneStepTimeSeriesSplit:
    '''Generate tuples of train_idx, test_idx pairs
    Assumes the index contains a level labeled "Date"'''

    def __init__(self, n_splits=3, test_period_length=1, shuffle=False):
        self.n_splits = n_splits
        self.test_period_length = test_period_length
        self.shuffle = shuffle
        self.test_end = n_splits * test_period_length


    @staticmethod
    def chunks(l, chunk_size):
        for i in range(0, len(l), chunk_size):
            yield l[i:i + chunk_size]

    def split(self, X, y=None, groups=None):
        unique_dates = (X.index
                       .get_level_values('Date')
                       .unique()
                       .sort_values(ascending=False)[:self.test_end])

        dates = X.reset_index()[['Date']]

        for test_date in self.chunks(unique_dates, self.test_period_length):
            train_idx = dates[dates.Date < min(test_date)].index
            test_idx = dates[dates.Date.isin(test_date)].index

            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# In[ ]:




