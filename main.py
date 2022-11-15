from portfolioOptimizer import Optimizer

assets = ['AAPL','PFE','INTC', 'XOM','CSGP','GIS','BAC','F','T', 'SNAP','CCL','HAS']

portfolio_optimizer = Optimizer(assets=assets, simulations=10000)
portfolio_optimizer.run()
