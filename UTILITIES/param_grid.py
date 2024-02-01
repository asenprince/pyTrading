__author__ = 'khalm prince'

'''this function declares parameter grid for GridSearchCV optimization'''

def get_param_grids(model='dt_reg'):

    if model=='dt_reg':
        param_grid = dict(
            max_depth = [1,2],
            min_samples_leaf= [10],
            max_features= [None, 'sqrt']
        )

    elif model=='dt_clf':
        param_grid = dict(
            max_depth= range(10, 20),
            min_samples_leaf=[250, 500, 750],
            max_features=['sqrt', 'log2']
        )


    elif model=='xgboost':
        param_grid = dict(
            learning_rate=[.01, .1, .3],
            max_depth=list(range(3, 14, 2)),
            colsample_bytree=[.8, 1],
            booster=['gbtree', 'dart'],
            gamma=[0, 1, 5],
        )


    elif model=='catboost':
        param_grid = dict(
            learning_rate=[.01, .1, .3],
            max_depth=list(range(3, 14, 2)),
            one_hot_max_size=[None, 2],
            max_ctr_complexity=[1, 2, 3],
            random_strength=[None, 1],
            colsample_bylevel=[.6, .8, 1]
        )

    elif model=='lightgbm':
        param_grid = dict(
            learning_rate=[.01, .1, .3],
            max_depth=list(range(3, 14, 2)),
            colsample_bytree=[.8, 1],
            max_bin=[32, 128],
            num_leaves=[2 ** i for i in range(9, 14)],
            boosting=['gbdt', 'dart'],
            min_gain_to_split=[0, 1, 5]
        )

    return param_grid
