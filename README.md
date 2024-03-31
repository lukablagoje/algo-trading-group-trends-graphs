# Algorithmic Trading Using Group Trends in Graph Representation
This project is a culmination of efforts by the WUTIS - Academic Trading And Investment Society, focusing on leveraging graph representations to understand and capitalize on group trends within stock market data. Our innovative approach, recognized through a first-place victory in an Algorithmic Trading pitch competition, involves creating a dynamic graph representation based on the cross-correlation of stock price time-series data, identifying coherent and deviating group trends among stocks.

Project Overview
The core of our project revolves around the following key components:

Graph Construction: We developed an algorithm to construct a graph representation where nodes represent individual stocks and edges denote the cross-correlation between stock prices over a sliding time window. This graph evolves over time, reflecting the changing dynamics of the market.

Trend Analysis: By analyzing the structure and evolution of the graph, we identified groups of stocks (subgraphs) that exhibit similar trends over time. This analysis allowed us to spot outlier stocks that deviate from their group's overall trend.

Signal Generation: Leveraging the insights from our graph analysis, we designed an algorithmic trading strategy that trades based on the expectation that outlier stocks will revert to their group's trend. This strategy includes rigorous backtesting over multiple market conditions to validate its effectiveness.

Backtesting Framework: Our backtesting infrastructure simulates the performance of our trading strategy using historical data, allowing us to refine our approach based on empirical evidence.
## Finding correlation groups in the training period:
![image](https://github.com/lukablagoje/quant_trading_networks/assets/52599010/53396e03-41d0-4ae3-a073-1f83fda918cc)

## Finding assests that deviate from group trends:
![image](https://github.com/lukablagoje/quant_trading_networks/assets/52599010/3ddf0297-719b-44d5-b28b-fc531d5f47ba)

 More details are in the [presentation file](https://github.com/lukablagoje/quant_trading_networks/blob/c83d7c6ddf2d7c90bacaa193ef0cf766ddb4efbd/network_representation_assets.pdf)
