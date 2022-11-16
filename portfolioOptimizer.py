import pandas as pd
import yfinance as yf
import numpy as np

import os
from tqdm import tqdm
import time
from itertools import combinations

from arch import arch_model


# Build optimizer
class Optimizer():
    def __init__(self, assets, interval='1wk', simulations=4000, maximize_return=True):
        # Define constants
        assets.sort()
        self.assets = assets
        self.SIMULATIONS = simulations
        self.COMBOS = list(combinations(assets, 2))
        self.maximize_return = maximize_return
        if interval=='1wk':
            self.N = 52
        else:
            self.N = 252
        
        # Gather asset data
        self.df = yf.download(tickers=assets, interval=interval)['Adj Close']
        self.df.ffill(inplace=True)
        self.df = np.log1p(self.df.pct_change())

        # Gather market data for beta calculations
        self.market = yf.download(tickers=assets+['^GSPC'], interval=interval)['Adj Close']
        self.market.ffill(inplace=True)
        self.market = np.log1p(self.market.pct_change())


    def garchModeling(self):
        garch = {}
        for asset in tqdm(self.assets):
            tseries = self.df[asset].dropna() * 100
            GARCH = arch_model(tseries, p=1,q=1, dist='t')
            output = GARCH.fit(disp='off')
            garch[asset] = output.conditional_volatility[-1] / 100

        return garch

    def optimize(self, garch):
        correlation_matrix = self.df.corr()
        portfolios = pd.DataFrame(
            columns=['Mu','Sigma','Sharpe','Beta'] + self.assets
        )

        for simulation in tqdm(range(self.SIMULATIONS)):
            WEIGHTS = np.random.dirichlet(np.ones(len(self.assets)),size=1).reshape(-1,)
            WEIGHTS = {a: b for a, b in zip(self.assets, WEIGHTS)}

            # Store weights
            portfolios.loc[simulation, self.assets] = np.round(pd.Series(WEIGHTS)*100,2)

            # Calculate portfolio expected return
            portfolio_return = np.sum(self.df.mean() * pd.Series(WEIGHTS))
            portfolio_return = (np.power((1+portfolio_return), self.N)-1) * 100
            portfolios.loc[simulation, 'Mu'] = np.round(portfolio_return, 2)

            # Begin calculating portfolio standard deviation
            portfolio_std = [WEIGHTS[_] * np.square(garch[_]) for _ in self.assets]

            # Further calculate portfolio standard deviation
            for combo in self.COMBOS:
                portfolio_std.append(
                    2 * WEIGHTS[combo[0]] * WEIGHTS[combo[1]] * correlation_matrix.loc[combo[0], combo[1]] * garch[combo[0]] * garch[combo[1]]
                )

            portfolio_std = np.sqrt(np.sum(portfolio_std))
            portfolio_std = (portfolio_std*np.sqrt(self.N))*100
            portfolios.loc[simulation, 'Sigma'] = np.round(portfolio_std, 2)

            # Calculate sharpe ratio
            sharpe = portfolio_return / portfolio_std
            portfolios.loc[simulation, 'Sharpe'] = np.round(sharpe, 2)

            # Calculate portfolio beta
            beta = np.sum(pd.Series(WEIGHTS) * np.array([self.market.cov().loc[asset,'^GSPC']/self.market['^GSPC'].var() for asset in self.assets]))
            portfolios.loc[simulation, 'Beta'] = np.round(beta,2)

        # Find the top 2.5% of portfolios
        if self.maximize_return:
            top_portfolios = portfolios.loc[(portfolios.Sharpe >= np.quantile(portfolios.Sharpe, 0.975)) & (portfolios.Mu >= np.quantile(portfolios.Mu, 0.975))]
        else:
            top_portfolios = portfolios.loc[(portfolios.Sharpe >= np.quantile(portfolios.Sharpe, 0.975)) & (portfolios.Sigma <= np.quantile(portfolios.Sigma, 0.025))]
        
        return top_portfolios


    def run(self, iterations=10):
        dataframe = pd.DataFrame()

        print('\n')
        print('Modeling data.')
        time.sleep(1)
        garch = self.garchModeling()

        print('\n')    
        print('Gathering portfolios.')
        time.sleep(1)
        for x in range(iterations):
            top_portfolios = self.optimize(garch)
            dataframe = pd.concat([dataframe, top_portfolios])

        print('\n')            
        print(f'Saving data to {os.getcwd()}')
        time.sleep(1)
        dataframe.to_csv('portfolios.csv',index=False)
        
        print('Done.')