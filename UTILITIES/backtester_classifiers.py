import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append("C:\\Users\\cyb3r53c\\Desktop\\Python Projects\\MYPROJECTS\\new\\Janson\\UTILITIES")
from gbm_utilities import get_data
from pathlib import Path

idx = pd.IndexSlice


class Backtester():
    ''' Class for the vectorized backtesting of simple Long-only Trading Strategies
    Attributes
    **************
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade
    
    Methods
    ***********
    get_data:
        imports the data
    
    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper)
    
    prepare_data:
        prepares the data for backtesting
        
    run_backtest:
        runs the strategy backtest
        
    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold
        
    optimize_strategy:
        backtests strategy for different parameter values incl. optimization and reporting (wrapper)
        
    find_best_strategy:
        finds the optimal strategy (global maximum)
    
    add_sessions:
        adds/labels trading sessions and their compound returns.
    
    add_leverage:
        adds leverage to the strategy
    
    print_performance:
        calculates and prints various performance metrics
    '''
    
    def __init__(self, symbol, features_list, model_list, tc, holding_period=1, start=None, end=None):
        
        self.symbol = symbol
        self.start = start
        self.end = end
        self.features = features_list[0]
        self.feature_type = features_list[1]
        self.estimator = model_list[0]
        self.modelName = model_list[1]
        self.data = None
        self.holding_period = holding_period
        self.DATASTORE = Path("C:\\Users\\cyb3r53c\\Desktop\\Python Projects\\MYPROJECTS\\new\\Janson\\data\\crypto_.h5")
        
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = None
        
    def __repr__(self):
        return "Backtester_with_{}_(symbol={})".format(self.modelName, self.symbol)
    
    def get_data(self):
        '''
        imports the data
        '''
        future_returns = get_data(start=self.start, end=self.end)[2].loc[self.symbol]
        processed_data = self.features.copy()
        processed_data['returns'] = future_returns.copy()
        self.data = processed_data
        
        
    def test_strategy(self):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (wrapper)
        
        Parameters
        ***********
        SMAs: tuple (SMA_S, SMA_M, SMA_L)
            Small, Medium and Large SMA to be used for the strategy
        '''
        
        self.prepared_data()
        self.run_backtest()

        data = self.results.copy()
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        
        #self.portfolioData.index = self.data.index

        self.print_performance()
    
    
    def prepared_data(self):
        ''' Prepares the Data for Backtesting'''
        self.get_data()
        df = self.data.copy()
        df['returns'] = df.returns.shift(self.holding_period)
        df = df.dropna()
        f = df.drop(columns=['returns'], axis=1)
        #print(f.columns)
        #print(f.head())
        df['estimate'] = self.estimator.predict(f)
        df['position'] = df.estimate-1 if self.modelName == 'xgboost' and 'long_short' in self.feaure_type else df.estimate
        self.tp_year = (df[df.columns[0]].count() / ((df.index.get_level_values('Date')[-1] - 
                                                     df.index.get_level_values('Date')[0]).days / 364))
        self.results = df

    def run_backtest(self):
        '''Runs the backtest'''
        data = self.results.copy()
        data['strategy'] = data['position'].shift(self.holding_period) * data['returns']
        data['trades'] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
         
        self.results = data
    
    # def simulateTrade(self):
    #     df = self.results.copy()
    #     inPosition = False
    #     entryPrice = 0 
    #     initialAmount = 100
    #     self.balance = initialAmount
    #     for index, row in df.iterrows():
    #         if not inPosition:
    #             if row['position'] == 1 or row['position'] == -1:
    #                 inPosition = True
    #                 entryPrice = row['Close']
                    
    #         elif inPosition and row['position'] == 1:
    #             if(row['Close'] >= 1.15 * entryPrice):
    #                 self.portfolioData.loc[index, 'Position'] = 'Sell'
    #                 self.portfolioData.loc[index, 'Exit'] = row['Close']
    #                 self.portfolioData.loc[index, 'PnL'] = 0.15 * initialAmount
    #                 self.portfolioData.loc[index, 'Trigger'] = 'TP'
    #                 inPosition = False
    #                 self.balance = self.balance + 0.15 * initialAmount
    #             elif row['Close'] <= 0.95 * entryPrice:
    #                 self.portfolioData.loc[index, 'Position'] = 'Sell'
    #                 self.portfolioData.loc[index, 'Exit'] = row['Close']
    #                 self.portfolioData.loc[index, 'PnL'] = -0.05 * initialAmount
    #                 self.portfolioData.loc[index, 'Trigger'] = 'SL'
    #                 inPosition = False
    #                 self.balance = self.balance - (0.05 * initialAmount)
    #             else:
    #                 continue
                        
    #         elif inPosition and row['position'] == -1:
    #             if(row['Close'] <= 0.70 * entryPrice):
    #                 self.portfolioData.loc[index, 'Position'] = 'Buy'
    #                 self.portfolioData.loc[index, 'Exit'] = row['Close']
    #                 self.portfolioData.loc[index, 'PnL'] = 0.15 * initialAmount
    #                 self.portfolioData.loc[index, 'Trigger'] = 'TP'
    #                 inPosition = False
    #                 self.balance = self.balance + 0.15 * initialAmount
    #             elif row['Close'] >= 1.05 * entryPrice:
    #                 self.portfolioData.loc[index, 'Position'] = 'Buy'
    #                 self.portfolioData.loc[index, 'Exit'] = row['Close']
    #                 self.portfolioData.loc[index, 'PnL'] = -0.05 * initialAmount
    #                 self.portfolioData.loc[index, 'Trigger'] = 'SL'
    #                 inPosition = False
    #                 self.balance = self.balance - (0.05 * initialAmount)
    #             else:
    #                 continue
                    
        
    def plot_results(self, leverage=False, path=None):
        ''' Plots the cummulative performance of the trading strategy compared to buy and hold'''
        
        if self.results is None:
            print("Run test_strategy() first")
        
        elif leverage:
            title = "MODEL = {} | SYMBOL = {} | FEATURE TYPE = {} | TC = {} | Leverage = {}".format(self.modelName, self.symbol, self.feature_type, self.tc, self.leverage)
            ax = self.results[['creturns', 'cstrategy', 'cstrategy_levered']].plot(title = title, figsize=(12, 8))
            if path is not None:
                plt.savefig(path)
        
        else:
            title = "MODEL = {} | {} | FEATURE TYPE = {}  | TC = {}".format(self.modelName, self.symbol, self.feature_type, self.tc)
            ax = self.results[['creturns', 'cstrategy']].plot(title = title, figsize=(12, 8))
            if path is not None:
                plt.savefig(path)
                
    def add_sessions(self, visualize=False):
        '''
        Adds/Labels Trading Sessions and their compound returns
        
        Parameter
        ***********
        Visualize: bool, default False
            if True, visualize compound session returns over time
        '''
        if self.results is None:
            print("Run test_strategy() first")
        
        data = self.results.copy()
        data['session'] = np.sign(data.trades).cumsum().shift().fillna(0)
        data['session_compound'] = data.groupby('session').strategy.cumsum().apply(np.exp) - 1
        self.results = data
        
        if visualize:
            data['session_compound'].plot(figsize=(12, 8))
            plt.show()
        
    def add_leverage(self, leverage, report=True): 
        '''
        Adds Leverage to the Strategy.

        Parameter
        ============
        leverage: float (positive)
            degree of leverage

        report: bool, default True
            if True, print Performance Report incl. leverage
        '''
        self.add_sessions()
        self.leverage = leverage

        data = self.results.copy()
        data['simple_ret'] = np.exp(data.strategy) - 1
        data['eff_lev'] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage )
        data.eff_lev.fillna(leverage, inplace=True)
        data.loc[data.trades != 0, 'eff_lev'] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data['strategy_levered'] = levered_returns
        data['cstrategy_levered']  = data.strategy_levered.add(1).cumprod()

        self.results = data

        if report:
            self.print_performance(leverage = True)

        ###################### Performance #################################

    
    def print_performance(self, leverage = False, display_results=False):
        ''' Calculates and prints various Performance Metrics
        '''
        
        data = self.results.copy()
        
        if leverage:
            to_analyze = np.log(data.strategy_levered.add(1))
        else:
            to_analyze = data.strategy
        
        self.strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        self.outperf = round(self.strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(to_analyze), 6)
        ann_mean = round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(self.calculate_annualized_std(to_analyze), 6)
        self.sharpe = round(self.calculate_sharpe(to_analyze), 6)
        cummulative_return = self.results.cstrategy[-1]
        if display_results:
            print(100 * "=")
            print(f"ML MODEL = {self.modelName} | INSTRUMENT = {self.symbol}_{self.feature_type}")
            print(100 * "-")
            print(f'Multiple (Strategy): {self.strategy_multiple} | Multiple (Buy and Hold): {bh_multiple} | Outperform: {self.outperf}')
            print(f'Cum Returns: {cummulative_return} | Sharpe Ratio: {self.sharpe} | Annualized Mean: {ann_mean} | CAGR: {cagr} | Annualized Volatility: {ann_std}')
        # print("PERFORMANCE MEASURES:")
        # print("\n")
        # print("Multiple (Strategy):             {}".format(self.strategy_multiple))
        # print("Multiple (Buy and Hold):         {}".format(bh_multiple))
        # print(38*"-")
        # print("Out-/Underperformance:           {}".format(self.outperf))
        # print("\n\n")
        # print("CAGR:                            {}".format(cagr))
        # print("Annualized Mean:                 {}".format(ann_mean))
        # print("Annualized Std:                  {}".format(ann_std))
        # print("Sharpe Ratio:                    {}".format(self.sharpe))
        
            print(100 * "=")
    
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
    
    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series):
        if series.std()==0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
    