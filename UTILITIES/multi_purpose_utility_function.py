__author__ = 'khalm_prince'

'''
    ############################ About ###############################
    A multi-purpose py file for performing various tasks
'''
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Model Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import Pool
from catboost import CatBoostClassifier
from itertools import product
from math import ceil
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz, _tree
import joblib
import warnings

# Metric Measurement Libraries
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, jaccard_score, r2_score, roc_auc_score, roc_curve


import sys, os
sys.path.append("C:\\Users\\cyb3r53c\\Desktop\\Python Projects\\MYPROJECTS\\new\\Janson\\UTILITIES")

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
idx = pd.IndexSlice

from gbm_utilities import format_time, get_data, get_datasets, get_dummies, factorize_cats, get_holdout_set, OneStepTimeSeriesSplit, get_assets

from param_grid import get_param_grids
from boosters_params import get_params
from backtester_classifiers import Backtester


model_parent_directory = Path('../DATA/MODELS/')
returns_images_directory = Path('../DATA/backtester_images/')
DATASTORE = Path("C:\\Users\\cyb3r53c\\Desktop\\Python Projects\\MYPROJECTS\\new\\Janson\\data\\crypto_.h5")

def create_directories(path):
    '''
        Creates Directory (if specified path does not exist)
        param:
            path -> str: the path to be created
        return:
            None
    '''
    
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as e:
            print(f"Error creating folder {path} | {e}")

cv = OneStepTimeSeriesSplit(n_splits=12) # initialized cross validation (cv) steps. to be used later

def format_value(x, r=2):

    '''
    formats float values to 2 decimal places
    params:
            x -> float: the float value to be formated
            r -> int: the decimal places (default: 2)

    return:
            formatted float value
    '''
    return round(x * 100, r)
# def get_reg_model_metrics(X, y, model):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
#     y_pred = model.predict(X_test)
#     #print(y_test)
#     mse = format_value(mean_squared_error(y_test, y_pred))
#     mae = format_value(mean_absolute_error(y_test, y_pred))
#     r2Score = format_value(r2_score(y_test, y_pred))
#     jScore = format_value(jaccard_score(y_test, y_pred))
#     return (mse, mae, r2Score, jScore)

def get_model(modelName, binary=False):
    ''' 
    a function to initialize a specified model
    params:
        modelName -> str : name of model to be initialized
        binary -> bool : treat output as multiclass or binary (True: binary; False: multiclass). Default: False
    returns:
        initialized model
    '''

    valid_modelList = ['xgboost', 'catboost', 'lightgbm', 'dt_clf']
    if modelName not in valid_modelList:
        return f'cannot handle {modelName} model'

    if modelName=='xgboost':
        model = XGBClassifier(enable_categorical=True, early_stopping_rounds=25)
    if modelName == 'catboost':
        model = CatBoostClassifier(allow_writing_files=False, early_stopping_rounds=25)
    if modelName == 'lightgbm':
        model = LGBMClassifier(objective='multiclass', early_stopping_rounds=25, verbose=-1) if not binary else LGBMClassifier(objective='binary', early_stopping_rounds=25, verbose=-1)
    if modelName == 'dt_clf':
        model = DecisionTreeClassifier()

    return model

def get_model_metrics(X_test, y_test, model, binary=True):
    ''' 
    a function to get model metrics
    params:
        X_test -> pandas DataFrame : Test Features
        y_test -> pandas Series / numpy array: Test Targets
        binary -> bool : treat output as multiclass or binary (True: binary; False: multiclass). Default: False
    
    returns:
        tuple of model metrics
    '''
    y_pred = model.predict(X_test)
    auc = np.nan
    if binary:
        auc = format_value(roc_auc_score(y_test, y_pred))
    accuracyScore = format_value(accuracy_score(y_test, y_pred))
    f1 = format_value(f1_score(y_test, y_pred, average='micro'))
    precision = format_value(precision_score(y_test, y_pred, average='micro'))
    recall = format_value(recall_score(y_test, y_pred, average='micro'))
    
    return (accuracyScore, f1, precision, recall, auc)

def build_and_optimize_model(start=False):
    ''' 
    a function to train, optimize, save, evaluate and backtest model
    params:
        start -> bool: start from beginning or continue from previous task
        
    returns:
        None
    '''

    if start:
        with pd.HDFStore(DATASTORE) as store:
            store.put('crypto/models/model_scores', pd.DataFrame()) ### initialize model scores (empty dataframe)
    
    with pd.HDFStore(DATASTORE) as store:
        model_data = store['crypto/models/model_scores'] # load model_data

    # create directories
    for directory in [model_parent_directory, returns_images_directory]:
        create_directories(directory) # create respective directories


    symbols = get_assets() # get symbols using get_assets library
    # model_data = pd.DataFrame(columns=['asset_details',
    #                                    'GridSearchScore',
    #                                    'AUC',
    #                                   'f1',
    #                                   'accuracy',
    #                                   'precision',
    #                                   'recall',
    #                                    'strategy_multiple',
    #                                    'outperf',
    #                                    'sharpe',
    #                                    'cum_returns',
    #                                    'num_of_trades'
    #                                   ]
                             # )

    long_short_dict = {
        'long_short': False,
        'long_only': True 
    }

    model_names = ['lightgbm', 'xgboost', 'catboost', 'dt_clf']

    # initialize dates
    model_data_end_date = pd.to_datetime('31-12-2022')
    end = '31-12-2022'
    backtest_start_date = pd.to_datetime('14-01-2023')
    
    symbols_with_inbalance_train_test_split = [] # a collector for unsuccessful assets 
    total = len(symbols) * len(long_short_dict) * 2
    print(f'total: {total}')
    #count=0
    for model_name in model_names:
        count=0
        print('='*100)
        print(model_name)
        print('-'*100)
        for binary_k, binary_v in long_short_dict.items():
            _, full_X, _ = get_data(dropna=True, binary=binary_v)
            model_y = get_data(end=end, dropna=True, binary=binary_v)[0]
            X_dummies = get_dummies(full_X)
            X_factors = factorize_cats(full_X)
            #backtest_y, backtest_X, _ = get_data(start=backtest_start_date, dropna=True, binary=binary_v)
            model_X_dummies = X_dummies.loc[idx[:, : model_data_end_date], :]
            backtest_X_dummies = X_dummies.loc[idx[:, backtest_start_date :], :]
            model_X_factors = X_factors.loc[idx[:, : model_data_end_date], :]
            backtest_X_factors = X_factors.loc[idx[:, backtest_start_date :], :]

            dataset = {
                'dummy': ((model_X_dummies, model_y), backtest_X_dummies),
                'factor': ((model_X_factors, model_y), backtest_X_factors)
            }

            #print(model_X.loc['1INCHUSDT'].info())
            
            for dk, dv in dataset.items():
                for symbol in symbols:
                    count+=1
                    asset_details = f'{symbol}_{model_name}_{dk}_{binary_k}'
                    filepath = f'{returns_images_directory}/{asset_details}.png'
                    if not os.path.exists(filepath):
                        print(count, end=' ', flush=True)
                        y_train, X_train, y_test, X_test = get_holdout_set(dv[0][1], dv[0][0], period=30*3)
                        #print(list(X_test.index.get_level_values('asset').unique()))
                        try:
                            X = dv[0][0].copy().loc[symbol]
                            y = dv[0][1].copy().loc[symbol]

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
                            b_X = dv[1].copy().loc[symbol]
                        
                        except Exception as e:
                            symbols_with_inbalance_train_test_split.append(symbol)
                        else:
                            gridsearchcv = GridSearchCV(estimator=get_model(model_name, binary_v),
                                                            cv=cv,
                                                           param_grid = get_param_grids(model_name),
                                                           n_jobs=-1,
                                                           refit=True,
                                                           return_train_score=True)
                            try:
                                gridsearchcv.fit(X=X_train, y=y_train+1) if model_name == 'xgboost' and not binary_v else gridsearchcv.fit(X=X_train, y=y_train)
                            
                            except Exception as e:
                                symbols_with_inbalance_train_test_split.append(symbol)
                            
                            else:
                                best_score = gridsearchcv.best_score_
                                best_score = format_value(best_score)
                                estimator = gridsearchcv.best_estimator_

                                metrics = get_model_metrics(X_test, y_test, model=estimator, binary=binary_v)

                                joblib.dump(estimator, Path(model_parent_directory, f'{asset_details}.joblib'))

                                btester = Backtester(symbol=symbol, features_list=(b_X, f'{dk}_{binary_v}'), model_list=(estimator, model_name), tc=-0.0005, start=backtest_start_date)
                                btester.test_strategy()
                                strategy_multiple = btester.strategy_multiple
                                outperformance = btester.outperf
                                sharpe = btester.sharpe
                                cumm_return = btester.results.cstrategy[-1]

                                number_of_trades = btester.results.position.value_counts()
                                num_trades = str(tuple(f'{index}: {value}' for index, value in zip(number_of_trades.index.astype(str), number_of_trades)))
                                # update model_data
                                new_data = {
                                    'asset_details': [asset_details],
                                    'GridSearchScore': best_score,
                                    'AUC': metrics[-1],
                                    'f1': metrics[1],
                                    'accuracy': metrics[0],
                                    'precision': metrics[2],
                                    'recall': metrics[3],
                                'strategy_multiple': strategy_multiple,
                                'outperf': outperformance,
                                'sharpe': sharpe,
                                'cum_returns': cumm_return,
                                'num_of_trades': num_trades}
                                new_df = pd.DataFrame(new_data).set_index('asset_details')
                                model_data = pd.concat([model_data, new_df])

                                btester.plot_results(leverage=False, path=Path(returns_images_directory, f'{asset_details}'))

                                with pd.HDFStore(DATASTORE) as store:
                                    store.put('crypto/models/model_scores', model_data)


                        finally:
                            if count == total:
                                print()
                                print(45* '*' + '  END  ' + '-'*45)

    print('Symbols not handled: ', *symbols_with_inbalance_train_test_split)