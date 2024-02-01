#!/usr/bin/env python
# coding: utf-8

# In[3]:
import pandas as pd
import numpy as np
from pathlib import Path

   # libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


DATASTORE = Path("C:\\Users\\cyb3r53c\\Desktop\\Python Projects\\MYPROJECTS\\new\\Janson\\data\\crypto_.h5")

def format_time(t):
    '''Return a formatted time string "HH:MM:SS" based on number time() value'''
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

def get_data(start=None, end=None, holding_period=1, dropna=True, binary=True, clf=True):
    'load dataset'
    valid_holding_periods=[1, 3, 7, 14, 30]

    trading_cost = 0.0005
    
    if holding_period not in valid_holding_periods:
        return f'cannot handle holding period {holding_period}'
    
    idx = pd.IndexSlice
    target = f'target_{holding_period}D'
    
    if start is not None:
        start = pd.to_datetime(start)
    if end is not None:
        end = pd.to_datetime(end)
        
    with pd.HDFStore(DATASTORE) as store:
        df = store['crypto/data'].loc[idx[:, start: end], :]
        
    if dropna:
        df = df.dropna()

    
    y_raw = df[target].copy()
    
    if clf:
        if binary:
            y = (y_raw>trading_cost*3).astype(int)
        
        else:
            values = list(np.where(y_raw > trading_cost, 1,
                        np.where(y_raw < -(trading_cost), -1, 0)))
        
            y = y_raw.map(dict(zip(y_raw, values)))
    else:
        if binary:
            values = list(np.where(y_raw > trading_cost, y_raw, 0))
            y = y_raw.map(dict(zip(y_raw, values)))
        
        else:
            y=y_raw
    cat_cols = ['month', 'weekday', 'day']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    X = df.drop([c for c in df.columns if c.startswith('target')], axis=1)
    #backtesting_data = X.copy()
    
    return y, X, y_raw

def save_y_raw():
    y_raw = get_data()[2]
    with pd.HDFStore(DATASTORE) as store:
        store.put('crypto/backtest_returns', y_raw)
        
def get_assets():
    return list(get_data(dropna=True)[0].index.get_level_values('asset').unique())

def get_dummies(df, cols=('month', 'weekday', 'day')):    
    cols = list(cols)
    df = pd.get_dummies(df,
                       columns=cols)
    return df

def factorize_cats(df, cats=('month', 'weekday', 'day')):
    #print(df.head())
    for cat in cats:
        df[cat] = pd.factorize(df[cat])[0]
    df.loc[:, cats] = df.loc[:, cats].fillna(-1).astype('category')
    return df


def get_holdout_set(target, features, period=6):
    #print(features)
    idx = pd.IndexSlice
    label = target.name
    dates = np.sort(target.index.get_level_values('Date').unique())
    cv_start, cv_end = dates[0], dates[-period - 2]
    holdout_start, holdout_end = dates[-period - 1], dates[-1]
    #print(features)
    df = features.join(target.to_frame())
    train = df.loc[idx[:, cv_start: cv_end], :]
    y_train, X_train = train[label], train.drop(label, axis=1)

    test = df.loc[idx[:, holdout_start: holdout_end], :]
    y_test, X_test = test[label], test.drop(label, axis=1)

    return y_train, X_train, y_test, X_test



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




def get_datasets(features, target, kfold, model='xgboost'):
    
    cat_cols = ['month', 'weekday', 'day', ]

    data = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        print(fold, end=' ', flush=True)
        if model == 'xgboost':
            data[fold] = {'train': xgb.DMatrix(label=target.iloc[train_idx],
                                              data=features.iloc[train_idx],
                                              nthread=-1),
                         'valid': xgb.DMatrix(label=target.iloc[test_idx],
                                              data=features.iloc[test_idx],
                                              nthread=-1)}
        elif model == 'lightgbm':
            train = lgb.Dataset(label=target.iloc[train_idx],
                               data=features.iloc[train_idx],
                               categorical_feature=cat_cols,
                               free_raw_data = False)
            # align validation data with histograms with training set
            valid = train.create_valid(label=target.iloc[test_idx],
                                      data=features.iloc[test_idx])

            data[fold] = {'train': train.construct(),
                         'valid': valid.construct()}


        elif model == 'catboost':
            # get categorical features
            cat_cols_idx = [features.columns.get_loc(c) for c in cat_cols]
            data[fold] = {'train': Pool(label=target.iloc[train_idx],
                                       data=features.iloc[train_idx],
                                       cat_features=cat_cols_idx),
                         'valid': Pool(label=target.iloc[test_idx],
                                      data=features.iloc[test_idx],
                                      cat_features=cat_cols_idx)}

    return data


def run_cv(test_params, data, n_splits=10, gb_machine='xgboost', binary=True):
    '''Train-Validate with early stopping'''
    #print('run_cv()', end=' ')
    if binary:
        scoring_metric_name = 'auc'
        scoring_metric = roc_auc_score
    
    else:
        scoring_metric_name = 'accuracy'
        scoring_metric = accuracy_score

    
    
    results = []
    cols = ['rounds', 'train', 'valid']
    for fold in range(n_splits):
        train = data[fold]['train']
        valid = data[fold]['valid']

        scores = {}
        if gb_machine == 'xgboost':
            model = xgb.train(params=test_params,
                             dtrain=train,
                             evals=list(zip([train, valid], ['train', 'valid'])),
                             verbose_eval = 50,
                             num_boost_round=250,
                             early_stopping_rounds=25,
                             evals_results=scores)
            
            results.append([model.best_iteration,
                           scores['train'][scoring_metric_name][-1],
                           scores['valid'][scoring_metric_name][-1]])

        elif gb_machine == 'lightgbm':
            #print('.', end=' | ')
            def custom_eval(y_true, y_pred, binary=True):
                if binary:
                    auc = roc_auc_score(y_true, y_pred)
                    if 'train' in scores and 'valid' in scores:
                        scores['valid']['custom_auc'].append(auc)
                        scores['train']['custom_auc'].append(auc)
                    elif 'train' not in scores and 'valid' not in scores:
                        scores['train'] = {'custom_auc': []}
                        scores['valid'] = {'custom_auc': []}
                        scores['train']['custom_auc'].append(auc)
                        scores['valid']['custom_auc'].append(auc)
                    return 'custom_auc', auc, True
                else:
                    acc = accuracy_score(y_true, y_pred)
                    if 'train' in scores and 'valid' in scores:
                        scores['valid']['custom_accuracy'].append(acc)
                        scores['train']['custom_accuracy'].append(acc)
                    elif 'train' not in scores and 'valid' not in scores:
                        scores['train'] = {'custom_accuracy': []}
                        scores['valid'] = {'custom_accuracy': []}
                        scores['train']['custom_accuracy'].append(acc)
                        scores['valid']['custom_accuracy'].append(acc)
                    return 'custom_accuracy', acc, True

            callbacks = [
                lgb.callback.early_stopping(stopping_rounds=25, first_metric_only=True)]
            model = lgb.train(params=test_params,
                             train_set=train,
                             valid_sets=[train, valid],
                             valid_names=['train', 'valid'],
                             num_boost_round=250,
                             feval=lambda preds, train_data: custom_eval(train_data.get_label(), preds),
                             callbacks=callbacks)       #print(scores)
            results.append([model.best_iteration,
                           scores['train'][f'custom_{scoring_metric_name}'][-1],
                           scores['valid'][f'custom_{scoring_metric_name}'][-1]])


        elif gb_machine=='catboost':
            model = CatBoostClassifier(**test_params)
            model.fit(X=train,
                     eval_set=[valid],
                     logging_level='Silent')
            train_score = model.predict_proba(train)[:, 1]
            valid_score = model.predict_proba(valid)[:, 1]
            
            results.append([
                model.tree_count_,
                scoring_metric(y_score=train_score, y_true=train.get_label()),
                scoring_metric(y_score=valid_score, y_true=valid.get_label())
            ])
            
    df = pd.DataFrame(results, columns=cols)


    result_mean = df.mean()
    result_std = df.std().rename({c: c + '_std' for c in cols})
    
    # Concatenate along rows (axis=0)
    #final_result_df = pd.concat([result_mean, result_std], ignore_index=True)
    final_result_df = pd.concat([result_mean, result_std, (pd.Series(test_params))], ignore_index=True)

    return final_result_df
    # return (df
    #         .mean()
    #         .to_frame()
    #         .append(df.std().rename({c: c + '_std' for c in cols}))
    #         .append(pd.DataFrame(test_params)))
