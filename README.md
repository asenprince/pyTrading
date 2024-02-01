# pyTrading
modules for python Trading

utility_functions.py (UTILITIES)
--------------------
for performing some utility functions. e.g.
#### get_history(client, symbol, interval, start, end=None) -> to retrieve historical candle data for an asset (symbol) from binance
#### live_keys(file): file is a text file that contains binance api key. live_keys() method retrieves the api keys (live) which serves to connect to binance
#### test_keys(file): same concept as live_keys(file) but used to retreive spot test api keys
#### futures_test_keys(file): same concept as live_keys(file) and test_keys(file). Reads and extracts futures test api keys

NB: For the methods retreiving api key to work, a text file containing your live keys and test keys should be stored in a specified data and passed to the function as file

feature_engineering.ipynb (notebooks)
------------------------
applied some feature engineering to the loaded instruments

multi_purpose_utility_function.py (UTILITIES)
---------------------------------------------
brings everything together (gbm_utilities.py, backtester_classifiers.py) to build, optimize and backtest the model

implementation.ipynb (notebooks)
--------------------------------
could take days running, depending on the number of assets (symbols) and number of models being trained. After running, a report is generated on the asset with the highest Model Score and the Highest Backtesting Returns

