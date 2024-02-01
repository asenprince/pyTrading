#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, precision_recall_curve,  accuracy_score
import numpy as np

def regression_benchmark(cv, X, y):
    rmse = []
    for train_idx, test_idx in cv.split(X):
        mean = y.iloc[train_idx].mean()
        data = y.iloc[test_idx].to_frame('y_test').assign(y_pred=mean)
        rmse.append(np.sqrt(mean_squared_error(data.y_test, data.y_pred)))
    return np.mean(rmse)


# In[3]:


def classification_benchmark(cv, X, y):
    acc_score = []
    for train_idx, test_idx in cv.split(X):
        mean = y.iloc[train_idx].mean()
        data = y.iloc[test_idx].to_frame('y_test').assign(y_pred=mean)
        acc_score.append(accuracy_score(data.y_test, data.y_pred))
    return np.mean(acc_score)


# In[ ]:




