# Algorithmic Trading Using Group Trends in Graph Representation
This project is part of a competition at WUTIS - Academic Trading And Investment Society, focusing on leveraging graph representations to understand and capitalize on group trends within stock market data. Our approach, recognized through a first-place victory in an Algorithmic Trading pitch competition, involves creating a dynamic graph representation based on the cross-correlation of stock price time-series data, identifying coherent and deviating group trends among stocks. More details are in the [presentation file](https://github.com/lukablagoje/quant_trading_networks/blob/c83d7c6ddf2d7c90bacaa193ef0cf766ddb4efbd/graph_representation_assets.pdf), with sample slides below:

### Finding group trends in the training period:
![image](https://github.com/lukablagoje/quant_trading_networks/assets/52599010/53396e03-41d0-4ae3-a073-1f83fda918cc)

### Finding assets that deviate from group trends:
![image](https://github.com/lukablagoje/quant_trading_networks/assets/52599010/3ddf0297-719b-44d5-b28b-fc531d5f47ba)
# Technical Project Overview
Our project is split into three sections:

[1. data_collection_graph_analysis.ipynb](https://github.com/lukablagoje/algo-trading-group-trends-graphs/blob/main/1.%20data_collection_graph_analysis.ipynb) - Collecting the data, constructing the representation graph and analyzing potential group parameters.

[2. group_trends_trading_strategy.ipynb](https://github.com/lukablagoje/algo-trading-group-trends-graphs/blob/main/2.%20group_trends_trading_strategy.ipynb) - Formulating a trading strategy around the stocks that deviate and potentially returning to group trends.

[3. parameter_optimization_strategy.ipynb] - Backtesting and ranking based on the historical data to find optimal parameters.

[trading_functions.py] - Python file where all used functions are stored.
## Finding correlation groups in the training period:
![image](https://github.com/lukablagoje/quant_trading_networks/assets/52599010/53396e03-41d0-4ae3-a073-1f83fda918cc)

# Data
![image](https://github.com/lukablagoje/quant_trading_networks/assets/52599010/3ddf0297-719b-44d5-b28b-fc531d5f47ba)

 More details are in the [presentation file](https://github.com/lukablagoje/quant_trading_networks/blob/c83d7c6ddf2d7c90bacaa193ef0cf766ddb4efbd/network_representation_assets.pdf)
